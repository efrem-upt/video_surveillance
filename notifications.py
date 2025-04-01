import requests
import logging
import threading

def send_push_notification(message, title="FCV - Alert!"):
    """
    Sends a push notification asynchronously via Pushover.
    """
    def send_notification():
        api_url = "https://api.pushover.net/1/messages.json"
        data = {
            "token": "<REPLACE_WITH_PUSHOVER_TOKEN>",  
            "user": "<REPLACE_WITH_PUSHOVER_USER>",  
            "message": message,
            "title": title,
        }
        try:
            response = requests.post(api_url, data=data)
            if response.status_code == 200:
                logging.info("Push notification sent successfully.")
            else:
                logging.error(f"Failed to send push notification: {response.text}")
        except Exception as e:
            logging.error(f"Error while sending push notification: {e}")
    
    thread = threading.Thread(target=send_notification)
    thread.daemon = True
    thread.start()