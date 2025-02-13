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
SAMPLE_RANGE_NAME = "input!A2:F"
SPREADSHEET_ID=""
MAX_COUNT = 99999
ACCEPTED_WORD_CLASSES = ['Adjective', 'Adverb', 'Pronoun', 'Preposition', 'Conjunction/subjunction', 'Interjection', 'Phrase', 'Noun', 'Verb']
ACCEPTED_FREQUENCIES = ['Everyday', 'Common', 'Rare', 'Archaic', 'Regional', 'Incorrect']

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

        columns=['Spanish', 'English', 'Swedish', 'Class', 'Eman', 'Sixten']
        df_in = pd.DataFrame(values, columns=columns).iloc[:MAX_COUNT]
        print(df_in)
        return df_in
    except HttpError as err:
        print(err)

def capitalize_first_alphanumeric(sentence):
    for i, char in enumerate(sentence):
        if char.isalnum():
            return sentence[:i] + char.upper() + sentence[i+1:]
    return sentence

def generate_examples(word_es, word, word_class):
    word_class = word_class.capitalize()
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'You complete missing CSV fields.'},
            {'role': 'user', 'content': f'Use the {word_class} \'{word_es}\' ({word}) in a simple Spanish (castellano) sentence.'},
            {'role': 'user', 'content': f'Then translate \'{word_es}\' to both <english> and <swedish>.' + (
                ' Use definite article adding \'the\' if missing.' if word_class == 'Noun' else
                ' Use infinitive form adding \'att\' and \'to\' if missing.' if word_class == 'Verb' else 
                f'''
                Also classify the <word_class> of \'{word_es}\' as one of:
                    - \'Adjective\',
                    - \'Adverb\',
                    - \'Pronoun\',
                    - \'Preposition\',
                    - \'Conjunction/subjunction\',
                    - \'Interjection\'.
                
                If none fits **leave it as \'Phrase\'**. <word_class> must be **exactly one word** (or two in the case of \'Conjunction/subjunction\').' if word_class == 'Phrase' else
                '''
            )},
            {
                'role': 'user',
                'content': f'''
                    Additionally determine the usage frequency of a phrase or word as one of:

                    - **Everyday** → Constantly used in everyday conversation and informal writing (e.g., "Hola", "O sea", "Sí"). Only give this category to extremely common words.
                    - **Common** → Frequently used but **less casual** or universal than "Everyday" (e.g., "Desarrollo", "Establecer").
                    - **Rare** → Slightly more advanced or technical etc. but still used (e.g., "Vituperar", "Desmonte").
                    - **Archaic** → Outdated of used very rarely, would sound out of place in conversation and writing. Mostly found in older literature or poetry (e.g., "Fuese", "Doquier", "Santo varón").
                    - **Regional** → Recognized, but mainly used in specific countries (e.g., "Chévere" in Venezuela, "Chamba" in Mexico).
                    - **Incorrect** → If \'{word_es}\' **is not a real Spanish word or phrase**. 

                    🚨 **IMPORTANT:**  
                    - Only give a word "Incorrect" in cases with clear grammar and spelling errors, or in cases where it has never been used. Otherwise **classify it as "Archaic"** instead of "Incorrect". Never classify a construct as "Incorrect" because it is infrequent or archaic. If the construct has never been used it is ok to classify it as "Incorrect".
                    - If unsure of the category, choose "Rare".
                    - Reply format must be **exactly one word**.
                '''
            },
            {'role': 'user', 'content': f'Provide a simple <sentence_in_spanish>, the <sentence_in_english> using \'{word_es}\', and a very brief english <comment> on usage of \'{word_es}\'. Avoid semicolons. Using only the \' symbol for quotes if they are necessary.'},
            {'role': 'user', 'content': 'It is highly important to **stick to the following format**:<english>|<swedish>|<word_class>|<sentence_in_spanish>|<sentence_in_english>|<frequency>|<comment>'}
        ]
    )

    # Get the content of the response
    content = response.choices[0].message.content

    # Initialize variables
    is_noun_or_verb = word_class != 'Noun' or word_class != 'Verb'
    _word_class = ''
    example_es = ''
    example_en = ''
    frequency = ''
    comment = ''

    # Split the content by lines and extract the relevant parts
    content_split = content.split('|')
    if len(content_split) != 7:
        print('Error:', content_split)

    word_en, word_sv, _word_class, example_es, example_en, frequency, comment = content_split

    word_class = word_class if is_noun_or_verb else _word_class
    word_class = re.sub(r'["\']+', '', word_class).strip().capitalize()
    word_class = word_class if word_class in ACCEPTED_WORD_CLASSES else 'Phrase'
    frequency = frequency.strip().capitalize() if frequency.strip().capitalize() in ACCEPTED_FREQUENCIES else 'Incorrect'

    return {
        'Spanish': capitalize_first_alphanumeric(word_es),
        'English': re.sub(r'["\']+', '', word_en).strip().capitalize(),
        'Swedish': re.sub(r'["\']+', '', word_sv).strip().capitalize(),
        'Class': word_class,
        'Example ES': re.sub(r'["\']+', '', example_es).strip(),
        'Example EN': re.sub(r'["\']+', '', example_en).strip(),
        'Frequency': frequency,
        'Comment': re.sub(r'["\']+', '', comment).strip(),
    }

def generate_frequencies(output_file):
    df_out = pd.read_csv(output_file, delimiter=';')
    count_added = 0
    batch_size = 20
    batch_number = 0
    offset = 0

    for index, row in df_out.iterrows():
        if offset <= index:         
            word_es = row['Spanish']
            word_en = row['English']
            word_se = row['Swedish']
            word_class = row['Class']

            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {
                        'role': 'system',
                        'content': f'''
                            You determine the usage frequency of a phrase or word as one of:

                            - **Everyday** → Constantly used in everyday conversation and informal writing (e.g., "Hola", "O sea", "Sí"). Only give this category to extremely common words.
                            - **Common** → Frequently used but **less casual** or universal than "Everyday" (e.g., "Desarrollo", "Establecer").
                            - **Rare** → Slightly more advanced or technical etc. but still used (e.g., "Vituperar", "Desmonte").
                            - **Archaic** → Outdated of used very rarely, would sound out of place in conversation and writing. Mostly found in older literature or poetry (e.g., "Fuese", "Doquier", "Santo varón").
                            - **Regional** → Recognized, but mainly used in specific countries (e.g., "Chévere" in Venezuela, "Chamba" in Mexico).
                            - **Incorrect** → If \'{word_es}\' **is not a real Spanish word or phrase**. 

                            🚨 **IMPORTANT:**  
                            - Only give a word "Incorrect" in cases with clear grammar and spelling errors, or in cases where it has never been used. Otherwise **classify it as "Archaic"** instead of "Incorrect". Never classify a construct as "Incorrect" because it is infrequent or archaic. If the construct has never been used it is ok to classify it as "Incorrect".
                            - If unsure of the category, choose "Rare".
                            - Reply format must be **exactly one word**.
                        '''
                    },
                    {
                        'role': 'user',
                        'content': f'What is the usage frequency for the \'{word_class}\' \'{word_es}\' meaning \'{word_en}\' in English and \'{word_se}\' in Swedish?'
                    }
                ]
            )

            frequency_parsed = response.choices[0].message.content.strip().capitalize()
            frequency = 'Incorrect' if frequency_parsed not in ACCEPTED_FREQUENCIES else frequency_parsed
            df_out.at[index, 'Tags'] = ' '.join(
                filter(None, [str(row['Tags']) if pd.notna(row['Tags']) else '', str(frequency).strip()])
            ).strip()
            count_added += 1
            print(f"{index}, Frequency for {row['Spanish']} is {frequency}")
        if (count_added + 1) % batch_size == 0:
            df_out.to_csv('output.csv', index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
            print(f'{index}: saving batch {batch_number}, modified {count_added}.')

    df_out.to_csv('output.csv', index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
    print(f'Completed {count_added} rows out of a total of {len(df_out)}.')

def generate_word_classes(output_file):
    df_out = pd.read_csv(output_file, delimiter=';')
    count_added = 0
    batch_size = 20
    batch_number = 0
    offset = 0

    for index, row in df_out.iterrows():
        if row['Class'] == 'Phrase' and offset <= index:         
            word_es = row['Spanish']
            word_en = row['English']
            word_se = row['Swedish']

            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': 'You classify a spanish phrase as one of: \'Adjective\', \'Adverb\', \'Pronoun\', \'Preposition\', \'Conjunction/subjunction\', \'Interjection\'. If none fits leave it as \'Phrase\'. Reply format must be exactly one word (or two in the case of \'Conjunction/subjunction\').'},
                    {'role': 'user', 'content': f'What is the class for the \'{word_es}\' meaning \'{word_en}\' in english and \'{word_se}\' in swedish?'},
                ]
            )

            word_class = 'Phrase' if response.choices[0].message.content not in ACCEPTED_WORD_CLASSES else response.choices[0].message.content
            df_out.at[index, 'Class'] = word_class
            count_added += 1
            print(f"{index}, Class for {row['Spanish']} is {word_class}")
        if (count_added + 1) % batch_size == 0:
            df_out.to_csv('output.csv', index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
            print(f'{index}: saving batch {batch_number}, modified {count_added}.')

    df_out.to_csv('output.csv', index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
    print(f'Completed {count_added} rows out of a total of {len(df_out)}.')

def process_csv(output_file):
    df_in = fetch_data()
    df_out = pd.read_csv(output_file, delimiter=';', index_col=False)
    batch_size = 10
    batch_number = 0
    offset = 0
    count_added = 0

    if df_out.empty:
        df_out = df_in.copy()
    else:
        df_in_lower = df_in.copy()
        df_out_lower = df_out.copy()
        df_in_lower['Spanish'] = df_in['Spanish'].str.lower()
        df_out_lower['Spanish'] = df_out['Spanish'].str.lower()
        diff_rows = df_in[~df_in_lower['Spanish'].isin(df_out_lower['Spanish'])].copy()

        # Replace "x" with the column name, keep empty otherwise
        df_in['Spanish'] = df_in['Spanish'].apply(capitalize_first_alphanumeric)
        df_in['Eman'] = df_in['Eman'].apply(lambda x: 'Eman' if x == 'x' else '')
        df_in['Sixten'] = df_in['Sixten'].apply(lambda x: 'Sixten' if x == 'x' else '')

        # Merge new rows and drop "Eman" and "Sixten" columns
        df_out = pd.concat([df_out, diff_rows], ignore_index=True).drop_duplicates(subset=['Spanish'], keep='first').drop(columns=['Eman', 'Sixten'], errors='ignore')
        
        # Map Tags correctly from df_in
        df_out['Tags_Tmp'] = ''
        df_out['Tags_Tmp'] = df_out['Spanish'].map(
            df_in.set_index('Spanish')[['Eman', 'Sixten']]
            .apply(lambda x: ' '.join(x[x != '']).strip(), axis=1)
        ).fillna(df_out['Tags_Tmp'])

    for index, row in df_out.iterrows():
        tags = str(df_out.at[index, 'Tags']).split(" ") if not pd.isna(df_out.at[index, 'Tags']) else []
        tags_set = set(tags)
        tags_set.discard('Eman')
        tags_set.discard('Sixten')
        tags = list(tags_set)
        tags_tmp = df_out.at[index, 'Tags_Tmp'].split(" ")

        if index < offset:
            continue

        is_new = pd.isna(row.get('Example ES'))

        if pd.isna(row['Spanish']):
            df_out.at[index, 'Error'] = 'Spanish required'
            continue

        word_es = row['Spanish']
        word_class = row['Class'] if not pd.isna(row['Class']) else 'Phrase'
        
        frequency = ''
        word_en = None
        word_sv = None
        
        # if is_new and not pd.isna(row['Spanish']) and word_class == 'Noun' and len(row['Spanish'].split(" ")) < 2:
        #     df_out.at[index, 'Error'] = f'{word_es}: Noun must have article'
        #     continue
    
        # if not pd.isna(row['Spanish']) and word_class == 'Verb' and (len(row['English'].split(" ")) < 2 or len(row['Swedish'].split(" ")) < 2):
        #     df_out.at[index, 'Error'] = f'{word_es}: Verb must be in infinitive form'
        #     continue

        if not pd.isna(row['English']) and len(row['English']):
            word_en = row['English'].capitalize().replace('_', ' ')
        elif not pd.isna(row['Swedish'] and len(row['Swedish'])):
            word_sv = row['Swedish'].capitalize().replace('_', ' ')
        else:
            df_out.at[index, 'Error'] = f'{word_es}: Swedish OR English required'
            continue
    
        if is_new:
            examples = generate_examples(word_es, word_en if word_en else word_sv, word_class)
            word_class = examples['Class']
            frequency = examples['Frequency']

            print(f"{examples['Spanish']} - {examples['Class']} - {examples['Example ES']} - {examples['Example EN']}")
            
            is_original_en = word_en is not None and len(word_en)
            is_original_sv = word_sv is not None and len(word_sv)

            df_out.at[index, 'Spanish'] = word_es
            df_out.at[index, 'English'] = f"{word_en if is_original_en else examples['English']}" + (' \u2713' if is_original_en else '')
            df_out.at[index, 'Swedish'] = f"{word_sv if is_original_sv else examples['Swedish']}" + (' \u2713' if is_original_sv else '')
            df_out.at[index, 'Class'] = examples['Class']
            df_out.at[index, 'Example ES'] = examples['Example ES']
            df_out.at[index, 'Example EN'] = examples['Example EN']
            df_out.at[index, 'Comment'] = examples['Comment']
            count_added += 1
        
        unique_tags = list(set(tags + tags_tmp + [word_class, frequency]))
        unique_tags.remove('')
        unique_tags.sort()
        df_out.at[index, 'Tags'] = ' '.join(filter(None, unique_tags)).strip()

        if (count_added + 1) % batch_size == 0:
            df_out.to_csv(output_file, index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
            print(f'{batch_number}-{index}: added {count_added} rows, out of a total of {len(df_out)}.')
            batch_number += 1

    df_out.drop(columns=['Tags_Tmp'], inplace=True)
    df_out.to_csv(output_file, index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)
    print(f'Completed {count_added} rows out of a total of {len(df_out)}.')

def normalize_csv(input_file, output_file, read_delimiter=';', usecols=[0], write_delimiter=';', skip_rows=0):
    df = pd.read_csv(input_file, delimiter=read_delimiter, index_col=False, skiprows=skip_rows, usecols=usecols, dtype=str)
    df[df.columns[0]] = df[df.columns[0]].str.lower()
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    df.to_csv(output_file, index=False, header=False, sep=write_delimiter, quoting=csv.QUOTE_MINIMAL)
    return df

def compare_dfs(df_1, df_2):
    set_output = set(df_1.iloc[:, 0])
    set_default = set(df_2.iloc[:, 0])

    words_in_output_not_default = set_output - set_default
    words_in_default_not_output = set_default - set_output

    for word in sorted(words_in_output_not_default):
        print(f"+{word}")

    for word in sorted(words_in_default_not_output):
        print(f"-{word}")

if __name__ == '__main__':
    process_csv('output.csv')
    # input = normalize_csv('input.csv', 'input_normalized.csv', read_delimiter=';', write_delimiter=';')
    # output = normalize_csv('output.csv', 'output_normalized.csv', read_delimiter=';', write_delimiter=';', usecols=['Spanish'])
    # default = normalize_csv('Default.txt', 'default_normalized.csv', read_delimiter='\t', write_delimiter=';', skip_rows=2)
    # compare_dfs(output, default)
    # generate_word_classes('output.csv')
    # generate_frequencies('output.csv')