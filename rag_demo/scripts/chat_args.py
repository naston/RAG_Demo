import argparse

def get_chat_mode():
    parser = argparse.ArgumentParser(
                    prog='RAG_Demo',
                    description='Chat with your document storage using Mixtral based RAG')

    parser.add_argument('--mode', type=str, default='chat', choices=["chat","ping"], help='')

    return parser.parse_args().mode