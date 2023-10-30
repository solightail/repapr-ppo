import json
import requests

def send_line(channel_token, user_id, text):
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {channel_token}'}
    post = {'to': user_id, 'messages': [{'type': 'text', 'text': text}]}
    req = requests.post(url, headers=headers, data=json.dumps(post))

    if req.status_code != 200:
        print(req.text)

send_line(
    "",
    "",
    "テストメッセージ from Python 3.11.4 / papr-ppo / test / line.py"
)