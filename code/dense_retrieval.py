import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from tqdm import trange
from arguments import cfg

import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler)
from retrieval_model import BertEncoder

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrieval:
    def __init__(
        self,
        tokenize_fn,
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
        model_checkpoint = cfg.model.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        if cfg.encoder.dense_train == True:
            self.p_encoder = BertEncoder.from_pretrained(model_checkpoint)
            self.q_encoder = BertEncoder.from_pretrained(model_checkpoint)
        else:
            if not os.path.exists(os.path.join(data_path, 'dense')):
                os.makedirs(os.path.join(data_path, 'dense'))
            p_path = os.path.join(data_path, 'dense/p_encoder-{}.pt'.format(cfg.encoder.load_encoder_path))
            q_path = os.path.join(data_path, 'dense/q_encoder-{}.pt'.format(cfg.encoder.load_encoder_path))
            self.p_encoder = BertEncoder.from_pretrained(p_path)
            self.q_encoder = BertEncoder.from_pretrained(q_path)
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
        pickle_name = cfg.encoder.embedding_name
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Traininig encoders")
            if cfg.encoder.dense_train == True:
                self.p_encoder, self.q_encoder = self.encoder_train(self.args, self.train_dataset, self.p_encoder, self.q_encoder)
                self.p_encoder.save_pretrained('/opt/ml/input/data/dense/p_encoder-{}.pt'.format(cfg.encoder.encoder_postfix))
                self.q_encoder.save_pretrained('/opt/ml/input/data/dense/q_encoder-{}.pt'.format(cfg.encoder.encoder_postfix))
            print("Build passage embedding")
            eval_batch_size = 8

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
                        
                    outputs = self.p_encoder(**p_inputs).to('cpu').numpy()
                    p_embs.extend(outputs)
            if cfg.data.faiss_gpu:
                self.p_embedding = p_embs
            else:
                self.p_embedding = np.array(p_embs)
            print(self.p_embedding.shape)

            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")
    
    def encoder_train(self, args, dataset, p_model, q_model):
        q_seqs = self.tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = self.tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')

        dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
        # Dataloader
        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
        
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        
        # Start training!
        global_step = 0
        
        p_model.zero_grad()
        q_model.zero_grad()
        torch.cuda.empty_cache()
        
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        
            for step, batch in enumerate(epoch_iterator):
                self.q_encoder.train()
                self.p_encoder.train()
                
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
            
                p_inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2]
                            }
                
                q_inputs = {'input_ids': batch[3],
                            'attention_mask': batch[4],
                            'token_type_ids': batch[5]}
                
                p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
                q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)
            
            
                # Calculate similarity score & loss
                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
            
                # target: position of positive samples = diagonal element 
                targets = torch.arange(0, args.per_device_train_batch_size).long()
                if torch.cuda.is_available():
                    targets = targets.to('cuda')
            
                sim_scores = F.log_softmax(sim_scores, dim=1)
            
                loss = F.nll_loss(sim_scores, targets)
            
                loss.backward()
                optimizer.step()
                scheduler.step()
                q_model.zero_grad()
                p_model.zero_grad()
                global_step += 1
                
                torch.cuda.empty_cache()

        return p_model, q_model

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
            if cfg.data.faiss_gpu: 
                p_emb = self.p_embedding
            else:
                p_emb = self.p_embedding.astype(np.float32)
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            if cfg.data.faiss_gpu: 
                res = faiss.StandardGpuResources()
            quantizer = faiss.IndexFlatL2(emb_dim)
            if cfg.data.faiss_gpu:
                index_ivf = faiss.IndexIVFFlat(
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

    def retrieve(
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
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
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

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            q_seqs = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            with torch.no_grad():
                self.q_encoder.eval()
                q_embs = self.q_encoder(**q_seqs).to('cpu').numpy()
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            p_embs_cuda = torch.Tensor(self.p_embedding).to('cuda')
            q_embs_cuda = torch.Tensor(q_embs).to('cuda')

        with timer("query ex search"):
            result = torch.matmul(q_embs_cuda, torch.transpose(p_embs_cuda, 0, 1))
        rank = torch.argsort(result, dim=1, descending=True).squeeze()
        doc_score = result.squeeze()[rank].tolist()
        doc_indices = rank.tolist()
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
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
        with timer("transform"):
            q_seqs = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            with torch.no_grad():
                self.q_encoder.eval()
                q_embs = self.q_encoder(**q_seqs).to('cpu').numpy()
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            p_embs_cuda = torch.Tensor(self.p_embedding).to('cuda')
            q_embs_cuda = torch.Tensor(q_embs).to('cuda')
        result = torch.matmul(q_embs_cuda, torch.transpose(p_embs_cuda, 0, 1))
        rank = torch.argsort(result, dim=1, descending=True).squeeze()
        return rank

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
            q_embs = self.q_encoder(**q_seqs).to('cpu').numpy()
        torch.cuda.empty_cache()
        
        q_embs = q_embs.astype(np.float32)
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
            if cfg.data.faiss_gpu:
                q_embs = self.q_encoder(**q_seqs)
            else:
                q_embs = self.q_encoder(**q_seqs).to('cpu').numpy()
        torch.cuda.empty_cache()
        
        if cfg.data.faiss_gpu:
            pass
        else:
            q_embs = q_embs.astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="/opt/ml/input/data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="/opt/ml/input/data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False,)

    retriever = DenseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
