from gotify import Gotify
import yaml

# get gotify url from secrets.yml
with open('secrets.yml') as file:
    secrets = yaml.safe_load(file)
gotify_url = secrets['gotify']['url']
gotify_token = secrets['gotify']['token']

# initialize gotify
gotify = Gotify(gotify_url, gotify_token)

# == Functions ==
def send_message(title, message, priority=5):
    gotify.create_message(title=title, message=message, priority=priority)

