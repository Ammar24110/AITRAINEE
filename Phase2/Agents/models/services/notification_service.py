from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import os 

class NotificationService:

    def send_notification(self, message: str):
        current = datetime.now()

        sender = os.getenv("EMAIL_SENDER")
        receiver = os.getenv("EMAIL_RECEIVER")
        password = os.getenv("EMAIL_PASSWORD")

        subject = "Task Notification"

        body = f"Time: {current}\n\n{message}"

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = receiver

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)

            print(f"[{current}] Email sent")

            return {
                "success": True,
                "message": "Email sent"
            }

        except Exception as e:
            print(f"[{current}] Email failed")

            return {
                "success": False,
                "message": str(e)
            }