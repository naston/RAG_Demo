import requests

def ping():
    response = requests.post('http://<http://127.0.0.1:5000/>', files=None)
    print(response)

def start_chat():
    while True:
        query = input('Type your message here: ')

        if query == 'exit':
            break
        
        elif query == 'cont':
            continue

        files = {
            'query': (None, query),
        }   

        response = requests.post('http://<http://127.0.0.1:5000/process_form>', files=files)

        print(response)