from datetime import datetime


class NotificationService:

    def send_notification(self, message: str):

        current = datetime.now()

        print(f"[{current}] Notification: {message}")