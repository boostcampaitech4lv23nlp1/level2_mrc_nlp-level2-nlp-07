import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
from .poly_encoder.lightning_model import LightningPCModel

import faiss
import numpy as np
import pandas as pd
from datasets import  Dataset, load_from_disk
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from tqdm import trange

import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class polyRetrieval:
    def __init__(
        self,
        tokenize_fn,
        cfg,
        data_path: Optional[str] = "/opt/ml/input/data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab
            data_path:
                데이터가 보관되어 있는 경로입니다.
            context_path:
                Passage들이 묶여있는 파일명입니다.
            data_path/context_path가 존재해야합니다.
        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        self.cfg = cfg
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.p_embedding = None  # get_passage_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.
        
        train_dataset = load_from_disk(os.path.join(data_path, 'train_dataset'))
        self.train_dataset = train_dataset['train']
        self.args = TrainingArguments(
                    output_dir="dense_retrieval",
                    evaluation_strategy="epoch",
                    learning_rate=cfg.encoder.lr,
                    per_device_train_batch_size=cfg.encoder.batch_size,
                    per_device_eval_batch_size=cfg.encoder.batch_size,
                    num_train_epochs=cfg.encoder.epoch,
                    weight_decay=cfg.encoder.weight_decay
                )
        model_checkpoint = cfg.encoder.model_name   # 'model_ckpt/0103_qa-last.ckpt'
        self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        if cfg.encoder.ckpt == True:
            model = LightningPCModel.load_from_checkpoint(model_checkpoint, tokenizer = self.tokenizer)
            self.p_encoder = model.pc_model.cand_encoder
            self.q_encoder = model.pc_model.cand_encoder
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()
    def get_passage_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = self.cfg.encoder.embedding_name
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Traininig encoders")
            if self.cfg.encoder.dense_train == True:
                self.p_encoder, self.q_encoder = self.encoder_train(self.args, self.train_dataset, self.p_encoder, self.q_encoder)
                self.p_encoder.save_pretrained('/opt/ml/input/data/dense/p_encoder-{}.pt'.format(self.cfg.encoder.encoder_postfix))
                self.q_encoder.save_pretrained('/opt/ml/input/data/dense/q_encoder-{}.pt'.format(self.cfg.encoder.encoder_postfix))

            print("Build passage embedding")
            eval_batch_size = 4

            # Construt dataloader
            valid_p_seqs = self.tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
            valid_dataset = TensorDataset(valid_p_seqs['input_ids'], valid_p_seqs['attention_mask'], valid_p_seqs['token_type_ids'])
            valid_sampler = SequentialSampler(valid_dataset)
            valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=eval_batch_size)

            # Inference using the passage encoder to get dense embeddeings
            p_embs = []

            with torch.no_grad():
                epoch_iterator = tqdm(valid_dataloader, desc="Iteration", position=0, leave=True)
                self.p_encoder.eval()
                for _, batch in enumerate(epoch_iterator):
                    batch = tuple(t.cuda() for t in batch)

                    p_inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2]
                                }
                        
                    outputs = self.p_encoder(p_inputs['input_ids'])
                    outputs = outputs[:,0,:].to('cpu').numpy()
                    p_embs.extend(outputs)
            if self.cfg.encoder.faiss_gpu:
                self.p_embedding = p_embs
            else:
                self.p_embedding = np.array(p_embs)
                print(self.p_embedding.shape)

            with open(emd_path, "wb") as file:
                print('file',file)
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:
        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.
        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = np.array(self.p_embedding)   #.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            if self.cfg.encoder.faiss_gpu: 
                res = faiss.StandardGpuResources()
            quantizer = faiss.IndexFlatL2(emb_dim)
            if self.cfg.encoder.faiss_gpu:
                index_ivf = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
                self.indexer = faiss.index_cpu_to_gpu(res, 0, index_ivf)
            else:
                self.indexer = faiss.IndexIVFScalarQuantizer(
                    quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
                )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.
        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]
        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        q_seqs = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            self.q_encoder.eval()
            q_embs = self.q_encoder(q_seqs['input_ids']).to('cpu').numpy()
        torch.cuda.empty_cache()
        
        # q_embs = q_embs.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_embs, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        q_seqs = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            self.q_encoder.eval()
            q_embs = self.q_encoder(q_seqs['input_ids']).to('cpu').numpy()
        torch.cuda.empty_cache()
        
        q_embs = q_embs.astype(np.float32)
        q_embs = q_embs[:,0,:]
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()