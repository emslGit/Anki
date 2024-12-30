from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI
import csv
import pandas as pd
import re
import os.path

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
SAMPLE_RANGE_NAME = "input!A2:D"
SPREADSHEET_ID="YOUR_ID"
MAX_COUNT = 99999

client = OpenAI()

def fetch_data():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=SPREADSHEET_ID, range=SAMPLE_RANGE_NAME)
            .execute()
        )
        values = result.get("values", [])

        if not values:
            print("No data found.")
            return

        columns=['Spanish', 'English', 'Swedish', 'Tags']
        df_in = pd.DataFrame(values, columns=columns).iloc[:MAX_COUNT]
        print(df_in)
        return df_in
    except HttpError as err:
        print(err)

def generate_link(word_es, tag):
    base_url = 'https://dle.rae.es/'
    if tag == 'Verb':
        return f'{base_url}{word_es}?m=form#conjugacion'
    elif tag == 'Noun':
        _word_es = word_es.split(" ")[1] if len(word_es.split(" ")) > 1 else word_es
        return f'{base_url}{_word_es}?m=form'
    elif tag in ['Adjective']:
        return f'{base_url}{word_es}?m=form'
    else:
        return 'No link'

def capitalize_first_alphanumeric(sentence):
    for i, char in enumerate(sentence):
        if char.isalnum():
            return sentence[:i] + char.upper() + sentence[i+1:]
    return sentence

def generate_examples(word_es, word, tags):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'You complete missing CSV fields.'},
            {'role': 'user', 'content': f'Use the {tags} \'{word_es}\' ({word}) in a simple Spanish (castellano) sentence.'},
            {'role': 'user', 'content': f'Translate {word_es} to both <english> and <swedish>.{' Use definite article adding \'the\' if missing.' if tags == 'Noun' else ''}{' Use infinitive form adding \'att\' and \'to\' if missing.' if tags == 'Verb' else ''}'},
            {'role': 'user', 'content': f'Provide a simple <sentence_in_spanish>, the <sentence_in_english> using \'{word_es}\', and a very brief english <comment> fun fact on its usage \'{word_es}\'. Avoid semicolons. Use \' for quotes.'},
            {'role': 'user', 'content': 'Format:<english>|<swedish>|<sentence_in_spanish>|<sentence_in_english>|<comment>'}
        ]
    )

    # Get the content of the response
    content = response.choices[0].message.content

    # Initialize variables
    example_es = ''
    example_en = ''
    comment = ''

    # Split the content by lines and extract the relevant parts
    word_en, word_sv, example_es, example_en, comment = content.split('|')

    return {
        'Spanish': capitalize_first_alphanumeric(word_es),
        'English': re.sub(r'["\']+', '', word_en).strip().capitalize(),
        'Swedish': re.sub(r'["\']+', '', word_sv).strip().capitalize(),
        'Example ES': re.sub(r'["\']+', '', example_es).strip(),
        'Example EN': re.sub(r'["\']+', '', example_en).strip(),
        'Comment': re.sub(r'["\']+', '', comment).strip(),
    }

def process_csv(output_file):
    df_in = fetch_data()
    df_out = pd.read_csv(output_file, delimiter=';')
    count_added = 0

    if df_out.empty:
        df_out = df_in.copy()
    else:
        df_in_lower = df_in.copy()
        df_out_lower = df_out.copy()
        df_in_lower['Spanish'] = df_in['Spanish'].str.lower()
        df_out_lower['Spanish'] = df_out['Spanish'].str.lower()
        diff_rows = df_in[~df_in_lower['Spanish'].isin(df_out_lower['Spanish'])].copy()
        df_out = pd.concat([df_out, diff_rows], ignore_index=True).drop_duplicates(subset=['Spanish'], keep='first')

    required_columns = ['Example ES', 'Example EN', 'Comment', 'Link']
    for column in required_columns:
        if column not in df_out.columns:
            df_out[column] = None

    for index, row in df_out.iterrows():
        if not pd.isna(row['Example ES']):
            continue
        
        if pd.isna(row['Spanish']):
            df_out.at[index, 'Report'] = 'Spanish required'
            continue

        word_es = capitalize_first_alphanumeric(row['Spanish'])
            
        if pd.isna(row['Tags']):
            df_out.at[index, 'Report'] = f'{word_es}: Tags required'
            continue

        word_en = None
        word_sv = None
        tags = row['Tags'].capitalize()

        if not pd.isna(row['Spanish']) and tags == 'Noun' and len(row['Spanish'].split(" ")) < 2:
            df_out.at[index, 'Report'] = f'{word_es}: Noun must have article'
            continue

        if not pd.isna(row['English']) and len(row['English']):
            word_en = row['English'].capitalize().replace('_', ' ')
        elif not pd.isna(row['Swedish'] and len(row['Swedish'])):
            word_sv = row['Swedish'].capitalize().replace('_', ' ')
        else:
            df_out.at[index, 'Report'] = f'{word_es}: Swedish OR English required'
            continue

        examples = generate_examples(word_es, word_en if word_en else word_sv, tags)
        print(examples)

        df_out['Example ES'] = df_out['Example ES'].astype('string')
        df_out['Example EN'] = df_out['Example EN'].astype('string')
        df_out['Comment'] = df_out['Comment'].astype('string')

        is_original_en = word_en is not None and len(word_en)
        is_original_sv = word_sv is not None and len(word_sv)

        df_out.at[index, 'Spanish'] = word_es
        df_out.at[index, 'English'] = f'{word_en if is_original_en else examples['English']}{' \u2713' if is_original_en else ''}'
        df_out.at[index, 'Swedish'] = f'{word_sv if is_original_sv else examples['Swedish']}{' \u2713' if is_original_sv else ''}'
        df_out.at[index, 'Tags'] = tags
        df_out.at[index, 'Example ES'] = examples['Example ES']
        df_out.at[index, 'Example EN'] = examples['Example EN']
        df_out.at[index, 'Comment'] = examples['Comment']
        df_out.at[index, 'Link'] = generate_link(word_es, tags)
        
        count_added += 1

    df_out.to_csv(output_file, index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
    print(f'Added {count_added} rows, for a total of {len(df_out)}.')

if __name__ == '__main__':
    process_csv('output.csv')