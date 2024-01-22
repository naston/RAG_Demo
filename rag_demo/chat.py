from .scripts import start_chat, ping, get_chat_mode

if __name__ == '__main__':
    args = get_chat_mode()

    if args.mode=='chat'
        start_chat()
    elif args.mode=='ping':
        ping()
    else:
        raise Exception