from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI
import numpy as np
import csv
import pandas as pd
import re
import os.path

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1wEjJpBgwV5LJdWpDxzeC_3u2rUVt3QQQGZXbJNMJSNs"
MAX_COUNT = 99999
ACCEPTED_WORD_CLASSES = [
    'Adjective',
    'Adverb',
    'Pronoun',
    'Preposition',
    'Conjunction/subjunction',
    'Interjection',
    'Phrase',
    'Noun',
    'Verb'
]

client = OpenAI()
creds = None


def generate_creds():
    global creds

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds


def fetch_data(columns, range):
    creds = generate_creds()

    try:
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=SPREADSHEET_ID, range=range)
            .execute()
        )
        values = result.get("values", [])

        if not values:
            print("No data found.")
            return

        return pd.DataFrame(values, columns=columns).iloc[:MAX_COUNT]
    except HttpError as err:
        print(err)


def post_data(df, range):
    creds = generate_creds()

    try:
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()

        # Convert DataFrame to list-of-lists

        data = df.values.tolist()
        body = {'values': data}

        # Update the spreadsheet with the provided data
        result = sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=range,
            valueInputOption="RAW",  # Change to "USER_ENTERED" if you need formatting
            body=body
        ).execute()

        print(f"{result.get('updatedCells')} cells updated.")
    except HttpError as err:
        print(err)


def capitalize_first_alphanumeric(sentence):
    for i, char in enumerate(sentence):
        if char.isalnum():
            return sentence[:i] + char.upper() + sentence[i + 1:]
    return sentence


def translate_output(target_language):
    df = fetch_data(['Spanish', 'English', 'Swedish', 'Class', 'Example ES', 'Example EN', 'Comment', 'Skip'], "output!A2:H")
    df = df[df['Skip'] != 'Yes']
    df.reset_index(drop=True, inplace=True)

    batch_size = 10
    offset = 0

    for index, row in df.iterrows():
        if index < offset:
            continue

        word_es = row['Spanish']
        word_en = row['English']
        word_sv = row['Swedish']
        word_class = row['Class']
        example_es = row['Example ES']
        example_en = row['Example EN']

        is_original_en = word_en.endswith('\u2713')

        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "system",
                    "content": "You complete missing CSV fields. Follow the output format exactly."
                },
                {
                    "role": "user",
                    "content": (
                        f"You are given the Spanish word '{word_es}', which is a {word_class}"
                        f"meaning {f'{word_en} in English' if is_original_en else {f'{word_sv} in Swedish'}}."
                        f"You are also given the sentence '{example_es}' in Spanish, which means '{example_en}' in English."
                        "please do the following:"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"1. Translate that word directly into {target_language}.\n"
                        f"2. Translate the sentence into {target_language}, preserving its original spanish meaning."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Please output the response in exactly the following format without any extra text:\n\n"
                        f"<word_in_{target_language}>|<sentence_in_{target_language}>"
                    )
                }
            ]
        )

        # Get the content of the response
        content = response.choices[0].message.content
        content_split = content.split('|')

        if len(content_split) != 2:
            print('Error:', content)

        word_translated, example_translated = content_split

        df.at[index, 'Translated'] = capitalize_first_alphanumeric(word_translated)
        df.at[index, 'Example T'] = capitalize_first_alphanumeric(example_translated)
        df.at[index, 'Tags'] = word_class

        if (index + 1) % batch_size == 0 or index == len(df) - 1:
            df.to_csv('output_translated.csv', index=False,
                      sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
            print(f'{index + 1} rows translated out of a total of {len(df)}.')

    df = df[['Spanish', 'Translated', 'Example ES', 'Example T', 'Tags']]

    post_data(df, 'output_translated!A2:E')


def generate_examples(word_es, word, word_class):
    word_class = word_class.capitalize()
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": (
                    "You complete missing CSV fields. Follow the output format exactly. "
                    "Make sure translations are accurately provided in the languages specified."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Given the Spanish word '{word_es}', which is a {word_class} meaning '{word}', "
                    "please complete the following tasks exactly as specified:"
                )
            },
            {
                "role": "user",
                "content": (
                    "1. Create a simple, natural sentence in Spanish (castellano) correctly using the word.\n"
                    "2. Translate that sentence directly into accurate and natural English. (English ONLY)\n"
                    "3. Translate that sentence directly into accurate and natural Swedish. (Swedish ONLY)\n"
                    "Ensure the translations precisely match their language instructions."
                )
            },
            {
                "role": "user",
                "content": (
                    "4. Classify the word's grammatical category exactly from these options: "
                    "Adjective, Adverb, Pronoun, Preposition, Conjunction/subjunction, Interjection, or Phrase. "
                    "If the word is a Noun or Verb, please strictly follow these rules:\n"
                    "- Nouns: Use the definite article ('the' in English, 'den/det' in Swedish).\n"
                    "- Verbs: Use infinitive form with 'to' (English) and 'att' (Swedish)."
                )
            },
            {
                "role": "user",
                "content": (
                    "5. Provide a concise comment in English explaining typical usage of the word in Spanish."
                )
            },
            {
                "role": "user",
                "content": (
                    "Output the response EXACTLY in the following format, with no extra text or annotations:\n\n"
                    "<word_in_english>|<word_in_swedish>|<word_class>|<sentence_in_spanish>|<sentence_in_english>|<comment>"
                )
            }
        ]
    )

    # Get the content of the response
    content = response.choices[0].message.content

    # Initialize variables
    is_noun_or_verb = word_class != 'Noun' or word_class != 'Verb'
    _word_class = ''
    example_es = ''
    example_en = ''
    comment = ''

    # Split the content by lines and extract the relevant parts
    content_split = content.split('|')
    if len(content_split) != 6:
        print('Error:', content_split)

    word_en, word_sv, _word_class, example_es, example_en, comment = content_split

    word_class = word_class if is_noun_or_verb else _word_class
    word_class = re.sub(r'["\']+', '', word_class).strip().capitalize()
    word_class = word_class if word_class in ACCEPTED_WORD_CLASSES else 'Phrase'

    return {
        'Spanish': capitalize_first_alphanumeric(word_es),
        'English': re.sub(r'["\']+', '', word_en).strip().capitalize(),
        'Swedish': re.sub(r'["\']+', '', word_sv).strip().capitalize(),
        'Class': word_class,
        'Example ES': re.sub(r'["\']+', '', example_es).strip(),
        'Example EN': re.sub(r'["\']+', '', example_en).strip(),
        'Comment': re.sub(r'["\']+', '', comment).strip(),
    }


def process_table():
    df_in = fetch_data(['Spanish', 'English', 'Swedish', 'Class', 'Skip'], "input!A2:E")
    df_out = fetch_data(['Spanish', 'English', 'Swedish', 'Class', 'Example ES', 'Example EN', 'Comment'], "output!A2:G")
    batch_size = 10
    batch_number = 0
    offset = 0
    count_added = 0
    count = 0

    df_in['Skip'] = df_in['Skip'].apply(lambda row: 'Yes' if row == 'x' else 'No')
    df_in['Spanish'] = df_in['Spanish'].apply(str.lower)
    df_out['Spanish'] = df_out['Spanish'].apply(str.lower)
    new_rows = df_in[~df_in['Spanish'].isin(df_out['Spanish'])].copy()
    df_out = pd.concat([df_out, new_rows], ignore_index=True).drop_duplicates(subset=['Spanish'], keep='first')
    df_out = df_out.drop(columns=['Skip'], errors='ignore')
    df_out = df_out.merge(df_in[['Spanish', 'Skip', 'English', 'Swedish']], on='Spanish', how='left')
    df_out = df_out.replace({np.nan: None})

    try:
        for index, row in df_out.iterrows():
            if index < offset:
                continue

            is_new = pd.isna(row.get('Example ES'))
            is_original_en = not pd.isna(row['English_y']) and len(row['English_y'])

            word_es = capitalize_first_alphanumeric(row['Spanish'])
            word_class = row['Class'] if not pd.isna(row['Class']) else 'Phrase'
            word_en = row['English_y'] if is_original_en else row['English_x']
            word_sv = row['Swedish_x'] if is_original_en else row['Swedish_y']

            if is_new:
                examples = generate_examples(word_es, word_en if is_original_en else word_sv, word_class)

                word_en = f"{word_en if is_original_en else examples['English']}"
                word_sv = f"{word_sv if not is_original_en else examples['Swedish']}"
                word_class = examples['Class']
                df_out.at[index, 'Example ES'] = examples['Example ES']
                df_out.at[index, 'Example EN'] = examples['Example EN']
                df_out.at[index, 'Comment'] = examples['Comment']
                count_added += 1

            df_out.at[index, 'English'] = capitalize_first_alphanumeric(word_en) + (' \u2713' if is_original_en else '')
            df_out.at[index, 'Swedish'] = capitalize_first_alphanumeric(word_sv) + ('' if is_original_en else ' \u2713')
            df_out.at[index, 'Tags'] = word_class
            df_out.at[index, 'Spanish'] = word_es

            if (count_added + 1) % batch_size == 0 or index == len(df_out) - 1:
                print(df_out[['Spanish', 'Example ES', 'Example EN']])
                df_out.to_csv('output.csv', index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
                print(f'{batch_number}-{index}: added {count_added} rows, out of a total of {len(df_out)}.')
                batch_number += 1

            count += 1

        print(f'Completed {count} adding {count_added} rows out of a total of {len(df_out)}.')
        df_out = df_out[['Spanish', 'English', 'Swedish', 'Class', 'Example ES', 'Example EN', 'Comment', 'Tags', 'Skip']]
        df_out.to_csv('output.csv', index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
        return df_out
    except Exception as e:
        print(f'Interrupted at row {count} out of a total of {len(df_out)}.')
        df_out = df_out[['Spanish', 'English', 'Swedish', 'Class', 'Example ES', 'Example EN', 'Comment', 'Tags', 'Skip']]
        df_out.to_csv(f'output.{count}.csv', index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
        raise e


def normalize_df(df, output_file, write_delimiter=';', skip_rows=0):
    df[df.columns[0]] = df[df.columns[0]].str.lower()
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    df.to_csv(output_file, index=False, header=False, sep=write_delimiter, quoting=csv.QUOTE_MINIMAL)
    return df


def normalize_file(input_file, output_file, read_delimiter=';', usecols=[0], write_delimiter=';', skip_rows=0):
    df = pd.read_csv(input_file, delimiter=read_delimiter, index_col=False, skiprows=skip_rows, usecols=usecols, dtype=str)
    df[df.columns[0]] = df[df.columns[0]].str.lower()
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    df.to_csv(output_file, index=False, header=False, sep=write_delimiter, quoting=csv.QUOTE_MINIMAL)
    return df


def compare_dfs(df_in, df_out):
    duplicates_in = get_duplicates(df_in)
    duplicates_out = get_duplicates(df_out)

    if not duplicates_in.empty:
        print("Found duplicates in input:")
        print(duplicates_in)

    if not duplicates_out.empty:
        print("Found duplicates in output:")
        print(duplicates_out)

    set_input = set(df_in.iloc[:, 0])
    set_output = set(df_out.iloc[:, 0])

    words_in_input_not_output = set_input - set_output
    words_in_output_not_input = set_output - set_input

    for word in sorted(words_in_input_not_output):
        print(f"+{word}")

    for word in sorted(words_in_output_not_input):
        print(f"-{word}")


def get_duplicates(df):
    dup_mask = df.duplicated(subset=df.columns[0], keep=False)
    return df[dup_mask]


def remove_duplicates(df):
    df = df.drop_duplicates(subset=df.columns[0], keep='first')
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    return df


if __name__ == '__main__':
    post_data(process_table(), 'output!A2:I')
    # translate_output('Mandarin')
    input = normalize_df(fetch_data(['Spanish'], "input!A2:A"), 'input_normalized.csv', write_delimiter=';')
    output = normalize_df(fetch_data(['Spanish'], "output!A2:A"), 'output_normalized.csv', write_delimiter=';')
    default = normalize_file('Default.txt', 'default_normalized.csv', read_delimiter='\t', write_delimiter=';', skip_rows=2)
    compare_dfs(output, input)
