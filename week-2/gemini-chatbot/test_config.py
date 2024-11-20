import configparser

prompts = configparser.ConfigParser()
prompts.read('prompts.env')

try:
    topic = prompts.get('TEMPLATES', 'TOPIC')
    number = prompts.get('TEMPLATES', 'NUMBER')
    print(f"Topic: {topic}, Number: {number}")
except configparser.NoSectionError as e:
    print(f"Error: {e}")
