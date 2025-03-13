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
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID="1wEjJpBgwV5LJdWpDxzeC_3u2rUVt3QQQGZXbJNMJSNs"
MAX_COUNT = 99999
ACCEPTED_WORD_CLASSES = ['Adjective', 'Adverb', 'Pronoun', 'Preposition', 'Conjunction/subjunction', 'Interjection', 'Phrase', 'Noun', 'Verb']

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
			return sentence[:i] + char.upper() + sentence[i+1:]
	return sentence

def translate_output(target_language):
	df = fetch_data(['Spanish', 'English', 'Swedish', 'Class', 'Example ES', 'Example EN', 'Comment', 'Skip'], "output!A2:H")
	df = df[df['Skip'] != 'Yes']
	df.drop(columns=['Swedish', 'Comment', 'Skip'], inplace=True)
	df.reset_index(drop=True, inplace=True)

	for index, row in df.iterrows():
		word_es = row['Spanish']
		word_en = row['English']
		word_class = row['Class']
		example_es = row['Example ES']
		example_en = row['Example EN']
		
		response = client.chat.completions.create(
			model='gpt-4o-mini',
			messages=[
				{'role': 'system', 'content': f'You translate words and phrases from spanish to {target_language}'},
				{'role': 'user', 'content': f'1. Translate the \'{word_es}\', which is a \'{word_class}\' meaning \'{word_en}\' in english, into {target_language}.'},
				{'role': 'user', 'content': f'2. Then translate the acompanying example sentence \'{example_es}\' meaning \'{example_en}\' into {target_language}.'},
				{'role': 'user', 'content': 'It is highly important to **stick to the following format**:<word_translated>|<example_translated>'}
			]
		)

		# Get the content of the response
		content = response.choices[0].message.content
		content_split = content.split('|')

		if len(content_split) != 2:
			print('Error:', content_split)

		word_translated, example_translated = content_split

		df.at[index, f'Translated'] = capitalize_first_alphanumeric(word_translated)
		df.at[index, f'Example T'] = capitalize_first_alphanumeric(example_translated)
		df.at[index, f'Tags'] = word_class

	df.drop(columns=['English', 'Example EN', 'Class'], inplace=True)
	df = df[['Spanish', 'Translated', 'Example ES', 'Example T', 'Tags']]

	post_data(df, f'output_translated!A2:E')

def generate_examples(word_es, word, word_class):
	word_class = word_class.capitalize()
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
					f"Given the Spanish word '{word_es}', which is a {word_class} meaning '{word}' in English, "
					"please do the following:"
				)
			},
			{
				"role": "user",
				"content": (
					"1. Create a simple sentence in Spanish (castellano) that appropriately uses the word.\n"
					"2. Translate that sentence directly into English.\n"
					"3. Translate that sentence directly into Swedish."
				)
			},
			{
				"role": "user",
				"content": (
					"4. Classify the word's grammatical category. Use one of the following exactly: "
					"Adjective, Adverb, Pronoun, Preposition, Conjunction/subjunction, Interjection, or Phrase. "
					"If the word is a Noun or Verb, please follow these rules:\n"
					"- For Nouns: Use the definite article (add 'the' if missing).\n"
					"- For Verbs: Use the infinitive form and add 'att' for Swedish and 'to' for English if missing."
				)
			},
			{
				"role": "user",
				"content": (
					"5. Provide a very brief comment in English on how the word is used in Spanish."
				)
			},
			{
				"role": "user",
				"content": (
					"Please output the response in exactly the following format without any extra text:\n\n"
					"<english>|<swedish>|<word_class>|<sentence_in_spanish>|<sentence_in_english>|<comment>"
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

	df_in_lower = df_in.copy()
	df_out_lower = df_out.copy()
	df_in_lower['Spanish'] = df_in['Spanish'].str.lower()
	df_out_lower['Spanish'] = df_out['Spanish'].str.lower()

	diff_rows = df_in[~df_in_lower['Spanish'].isin(df_out_lower['Spanish'])].copy()
	diff_rows['Skip'] = diff_rows['Skip'].apply(lambda row: 'No' if pd.isna(row) else 'Yes')
	diff_rows['Original EN'] = diff_rows['English'].apply(capitalize_first_alphanumeric)
	diff_rows['Original SV'] = diff_rows['Swedish'].apply(capitalize_first_alphanumeric)

	df_out['Spanish'] = df_in['Spanish'].apply(capitalize_first_alphanumeric)
	df_out['Skip'] = df_in['Skip'].apply(lambda row: 'No' if pd.isna(row) else 'Yes')
	df_out['Original EN'] = df_in['English'].apply(capitalize_first_alphanumeric)
	df_out['Original SV'] = df_in['Swedish'].apply(capitalize_first_alphanumeric)
	df_out = pd.concat([df_out, diff_rows], ignore_index=True).drop_duplicates(subset=['Spanish'], keep='first')

	for index, row in df_out.iterrows():
		if index < offset:
			continue

		is_new = pd.isna(row.get('Example ES'))
		is_original_en = not pd.isna(row['Original EN']) and len(row['Original EN'])

		word_es = row['Spanish']
		word_class = row['Class'] if not pd.isna(row['Class']) else 'Phrase'
		word_en = (row['Original EN'] + ' \u2713') if is_original_en else row['English']
		word_sv = (row['Original SV'] + ' \u2713') if not is_original_en else row['Swedish']

		if is_new:
			examples = generate_examples(word_es, word_en if word_en else word_sv, word_class)

			# print(f"{examples['Spanish']} - {examples['English']} - {examples['Class']} - {examples['Example ES']} - {examples['Example EN']}")

			word_en = f"{word_en if is_original_en else examples['English']}"
			word_sv = f"{word_sv if not is_original_en else examples['Swedish']}"
			word_class = examples['Class']
			df_out.at[index, 'Example ES'] = examples['Example ES']
			df_out.at[index, 'Example EN'] = examples['Example EN']
			df_out.at[index, 'Comment'] = examples['Comment']
			count_added += 1
		
		df_out.at[index, 'English'] = word_en
		df_out.at[index, 'Swedish'] = word_sv
		df_out.at[index, 'Tags'] = word_class
		df_out.at[index, 'Spanish'] = word_es

		if (count_added + 1) % batch_size == 0 or index == len(df_out) - 1:
			df_out.to_csv('output.csv', index=False, sep=';', quotechar='"', quoting=csv.QUOTE_ALL)	
			print(f'{batch_number}-{index}: added {count_added} rows, out of a total of {len(df_out)}.')
			batch_number += 1

	df_out.drop(columns=['Original EN', 'Original SV'], inplace=True)	
	print(f'Completed {count_added} rows out of a total of {len(df_out)}.')
	return df_out

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

if __name__ == '__main__':
	# post_data(process_table(), 'output!A2:I')
	translate_output('Mandarin')
	# input = normalize_df(fetch_data(['Spanish'], "input!A2:A"), 'input_normalized.csv', write_delimiter=';')
	# output = normalize_df(fetch_data(['Spanish'], "output!A2:A"), 'output_normalized.csv', write_delimiter=';')
	# default = normalize_file('Default.txt', 'default_normalized.csv', read_delimiter='\t', write_delimiter=';', skip_rows=2)
	# compare_dfs(input, output)