import argparse
import requests
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
import pandas as pd


PROMPT = """/no_think 당신은 친절하고 전문적인 업무 지원 챗봇입니다.
아래 문맥정보를 바탕으로 질문에 한국어로 답변하십시오. 사용자가 이해하기 쉬운 단어를 사용해야 하며, 문맥정보에 없는 내용은 사용하지 마십시오.

## 문맥정보
{context}

## 사용자 질문
{query}"""


def load_directory(directory):
    reader = SimpleDirectoryReader(input_dir=directory)
    documents = reader.load_data()
    return documents


def load_file(file):
    reader = SimpleDirectoryReader(input_files=[file])
    documents = reader.load_data()
    return documents


def check_ollama_health(ollama_url):
    response = requests.get(ollama_url)
    print(response.text)
    return True if response.status_code == 200 else False


def get_model_list(ollama_url):
    response = requests.get(f"{ollama_url}/api/tags")
    models = response.json()
    return models["models"]


def model_exists(ollama_url, model):
    models = get_model_list(ollama_url)
    for model_ in models:
        if model_["name"] == model:
            return True
    return False


def print_model_list(ollama_url):
    models = get_model_list(ollama_url)
    print(f"{'모델명':<10} {'수정일자':<20} {'모델사이즈':<10}")
    for model in models:
        print(f"{model['name']:<10} {model['modified_at']:<20} {model['size']:<10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, dest="file_path")
    parser.add_argument("-d", "--directory", type=str, dest="directory", default="./data")
    parser.add_argument("-m", "--model", type=str, dest="model", default="qwen3:1.7b")
    parser.add_argument("-u", "--ollama_url", type=str, dest="ollama_url", default="http://0.0.0.0:11434")
    parser.add_argument("-e", "--evaluate_file_path", type=str, dest="evaluate_file_path", default="./questions.csv")

    args = parser.parse_args()

    if args.file_path is None and args.directory is None:
        raise ValueError("-f 또는 -d로 파일경로나 디렉토리 경로 지정 필요")

    if args.directory:
        documents = load_directory(args.directory)
    elif args.file_path:
        documents = load_file(args.file_path)

    if not check_ollama_health(args.ollama_url):
        raise ConnectionError("Ollama not running")

    if not model_exists(args.ollama_url, args.model):
        print_model_list(args.ollama_url)
        raise ValueError("-m 모델 추가 필요")

    llm = Ollama(base_url=args.ollama_url,
                 model=args.model,
                 request_timeout=300)

    df = pd.read_csv(args.evaluate_file_path)

    for idx, (page, question) in enumerate(df.values, 1):
        prompt = PROMPT.format(
            context=documents[page-1].text, query=question
        )
        print(f"({idx})--------------------------------------")
        print(f"Q: {question}")
        print(f"A: {llm.complete(prompt)}")
