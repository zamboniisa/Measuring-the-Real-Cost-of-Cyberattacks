import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from io import StringIO
import os
from google import genai
from google.genai import types
import base64
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import random
import time
import nest_asyncio
import asyncio
from playwright.async_api import async_playwright
import os
import google.generativeai as genai
import ast
import re
from typing import List, Dict
import os
import time
import calendar
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from time import sleep
from vertexai.generative_models import GenerativeModel, SafetySetting
import vertexai
import re
import json
import ast

path = '/Users/bibbi/Desktop/tesi isa/tesi-isa-1-d71a66bbbd87.json'

def chunk_retry(BASE_URL, params):
    resp = requests.get(BASE_URL, params=params)
    if resp.status_code != 200 or not resp.text.strip():
        return None
    return resp


def download_csv(name, date, days_param, try_ = False):
    # --- parse "Aug 2022" into numeric month/year ---
    try:
        mon_str, yr_str = date.split()
        month = datetime.strptime(mon_str, "%b").month
        year  = int(yr_str)
    except Exception:
        raise ValueError(f"date must be 'Mon YYYY', got {date!r}")

    # --- compute previous month/year and next month/year ---
    if month == 1:
        prev_month, prev_year = 12, year - 1
    else:
        prev_month, prev_year = month - 1, year

    if month == 12:
        next_month, next_year = 1, year + 1
    else:
        next_month, next_year = month + 1, year

    # --- build window ---
    # start at the 15th of the previous month
    period_start = datetime(prev_year, prev_month, 15, 0, 0)
    # end at the last day of the following month
    last_day     = calendar.monthrange(next_year, next_month)[1]
    period_end   = datetime(next_year, next_month, last_day, 0, 0)

    # --- GDELT parameters & keyword list ---
    BASE_URL    = "https://api.gdeltproject.org/api/v2/doc/doc"
    FORMAT      = "CSV"
    MODE        = "artList"
    MAXRECORDS  = 250

    # Tutti i termini relativi al cyber
    keywords = [# Termini di base
                "cyberattack", "keylogger", "cyber", "malicious", "malware", "cybercrime",
                "phishing", "vishing", "backdoor", "hacktivist", "unpatched", "bug",
                "firewall", "rootkit", "untrusted", "flaw", "vulnerability", "eavesdropping",
                "cybersecurity", "router", "hacker", "unsecure", "spoofing", "sniffing",
                "sanitization", "exploit", "disinformation", "malinformation",
                "ransomware", "data breach", "data leakage",

                # Tipi di attacco avanzati 
                "DDos", "denial of service", "man-in-the-middle", "MITM",
                "zero-day", "zero day", "SQL injection", "cross-site scripting", "XSS",
                "privilege escalation", "credential stuffing", "password spraying",
                "brute force", "dictionary attack", "session hijacking",
                "DNS poisoning", "ARP poisoning", "cache poisoning",
                "pharming", "watering hole", "water holing",
                "pass-the-hash", "spear phishing", "whaling", "smishing",

                # Malware e payload
                "trojan", "worm", "virus", "spyware", "adware", "cryptojacking",
                "botnet", "botnets", "ransomworm",

                # Infrastrutture e tattiche di rete
                "command and control", "C2", "honeypot", "honeynet",
                "pivoting", "lateral movement", "data exfiltration",

                # Ingegneria sociale
                "social engineering", 

                # Difesa e monitoraggio
                "exploit kit", "patch management", "incident response",
                "SOC", "threat hunting", "SIEM",

                # Altri concetti correlati
                "deepfake", "credential harvesting", "skimming", "card skimming"]

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # --- slide a 1-day window across the 3-month period ---
    start_time = period_start
    tot = 0

    if try_:
        start_time = datetime(prev_year, month, 15, 0, 0)
        print(f'Trying for {days_param} for company {name}')
        end_time = start_time + timedelta(days_param)

        start_str = start_time.strftime("%Y%m%d%H%M%S")
        end_str   = end_time.strftime("%Y%m%d%H%M%S")

        # query each small keyword chunk (to stay under 250 records)
        number_words = 2
        for group in tqdm(chunks(keywords, number_words)):
            if number_words == 1:
                query  = f'repeat5:"{name}" AND "{group[0]}"'
            else:
                quoted = [f'"{k}"' for k in group]
                joined_keywords = " OR ".join(quoted)
                query = f'repeat5:"{name}" AND ({joined_keywords})'

            # print(query)
            
            params = {
                "query":         query,
                "mode":          MODE,
                "format":        FORMAT,
                "maxrecords":    MAXRECORDS,
                "startdatetime": start_str,
                "enddatetime":   end_str
            }

            # maybe sleep

            resp = requests.get(BASE_URL, params=params)
            if resp.status_code != 200 or not resp.text.strip():
                print(f'Error Status {resp.status_code}')
                continue
            
            try:
                df = pd.read_csv(StringIO(resp.text))
                tot_local = len(df)
                # print(f'Number of articles found in this chunk {tot_local}')
                tot += tot_local
            except:
                # print(f'Number of articles found in this chunk 0')
                continue

        print(f'Number of articles found in this date range: {tot}')
        return tot
    
    print(f"Fetching for company {name} from {period_start:%Y-%m-%d} to {period_end:%Y-%m-%d}")
    combined_all = pd.DataFrame()
    count = 0
    start_clock = time.time()
    while start_time < period_end:
        end_time = start_time + timedelta(days_param)
        if end_time > period_end:
            end_time = period_end

        start_str = start_time.strftime("%Y%m%d%H%M%S")
        end_str   = end_time.strftime(  "%Y%m%d%H%M%S")

        print(f"[{start_time:%Y-%m-%d}] querying {start_str} → {end_str}")
        all_batches = []

        # query each small keyword chunk (to stay under 250 records)
        number_words = 2
        for group in tqdm(chunks(keywords, number_words)):
            count +=1
            if number_words == 1:
                query  = f'repeat5:"{name}" AND "{group[0]}"'
            else:
                quoted = [f'"{k}"' for k in group]
                joined_keywords = " OR ".join(quoted)
                query = f'repeat5:"{name}" AND ({joined_keywords})'

            params = {
                "query":         query,
                "mode":          MODE,
                "format":        FORMAT,
                "maxrecords":    MAXRECORDS,
                "startdatetime": start_str,
                "enddatetime":   end_str
            }

            # maybe sleep
            sleep(1)

            resp = requests.get(BASE_URL, params=params)
            if resp.status_code != 200 or not resp.text.strip():
                print(f'Error Status {resp.status_code}!')
                if resp.status_code == 429:
                    elapsed = time.time() - start_clock
                    print(f'Elapsed time before 429 error : {elapsed}')
                    print(f'Number of calls before 429 error : {count}')
                retry = None
                print('Retrying for Error 429!')
                while retry is None:
                    retry = chunk_retry(BASE_URL, params)
                print('Error 429 passed!')
                try:
                    df = pd.read_csv(StringIO(retry.text))
                    #print(f'Number of articles found in this chunk: {len(df)}')
                    all_batches.append(df)
                except Exception as e:
                    pass
                    #print(f'Number of articles found in this chuck: 0')
                    #print(f'Error: {e}')
                count = 0
                start_clock = time.time()
                continue

            try:
                df = pd.read_csv(StringIO(resp.text))
                #print(f'Number of articles found in this chunk: {len(df)}')
                all_batches.append(df)
            except Exception as e:
                pass
                #print(f'Number of articles found in this chuck: 0')
                #print(f'Error: {e}')
            
        start_time = end_time

        try:
            combined_date = pd.concat(all_batches, ignore_index=True)
            print(f'Number of articles found in this date range: {len(combined_date)}')
            combined_all = pd.concat([combined_all, combined_date], ignore_index=True)
        except Exception as e:
            print(f'Number of articles found in this date range: 0')
            print(f'Error: {e}')

    try:
        os.makedirs(f"data/{name}", exist_ok=True)
        out_path = f"data/{name}/gdelt_{date.replace(' ', '_')}.csv"
        combined_all.to_csv(out_path, index=False)
        print(f"Saved {len(combined_all)} total articles to {out_path}")
        print(f'Number of articles found for {name}: {len(combined_all)}')
    except Exception as e:
        print(f'Number of articles found for {name}: 0')
        print(f'Error: {e}')


def generate(msg, prompt):
    # Initialise Vertex AI
    vertexai.init(
        project="tesi-isa-1",
        location="global"
    )

    model = GenerativeModel("gemini-2.5-flash")

    safety_settings = [
        SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]

    # Combine prompt with system instruction
    system_instruction = f"{prompt}"
    user_message = msg

    # Generate content
    responses = model.generate_content(
        [system_instruction, user_message],
        safety_settings=safety_settings,
        generation_config={
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": 60000,
            "seed": 0
        },
        stream=True  # Equivalent to your `generate_content_stream`
    )

    full_response = ""
    for chunk in responses:
        if chunk.candidates and chunk.candidates[0].content.parts:
            full_response += chunk.candidates[0].content.parts[0].text

    return full_response


# def generate(msg, prompt):
#   client = genai.Client(
#       vertexai=True,
#       project="gen-lang-client-0273914738",
#       location="global",
#   )

#   msg1_text1 = msg

#   si_text1 = prompt
#   model = "gemini-2.5-flash"
#   contents = [
#     types.Content(
#       role="user",
#       parts=[
#         types.Part(text=msg1_text1)
#       ]
#     ),
#   ]

#   generate_content_config = types.GenerateContentConfig(
#     temperature = 0,
#     top_p = 1,
#     seed = 0,
#     max_output_tokens = 60000,
#     safety_settings = [types.SafetySetting(
#       category="HARM_CATEGORY_HATE_SPEECH",
#       threshold="OFF"
#     ),types.SafetySetting(
#       category="HARM_CATEGORY_DANGEROUS_CONTENT",
#       threshold="OFF"
#     ),types.SafetySetting(
#       category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
#       threshold="OFF"
#     ),types.SafetySetting(
#       category="HARM_CATEGORY_HARASSMENT",
#       threshold="OFF"
#     )],
#     system_instruction=[types.Part.from_text(text=si_text1)],
#   )

#   full_response = ""
#   for chunk in client.models.generate_content_stream(
#       model="gemini-2.5-flash",
#       contents=contents,
#       config=generate_content_config,
#   ):
#     full_response += chunk.text

#   return full_response


# Fix asyncio loop issues (needed for Jupyter or nested loops)
nest_asyncio.apply()

# USER AGENT ROTATION
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.86 Mobile Safari/537.36"
]

# FAST: BASIC EXTRACTION VIA REQUESTS
def extract_with_requests(url):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://www.google.com"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split())
    except Exception:
        return None  # Trigger fallback

# STRONG: ASYNC PLAYWRIGHT FALLBACK
async def extract_with_browser_async(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
            page = await context.new_page()
            await page.goto(url, timeout=20000)
            html = await page.content()
            await browser.close()

        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split())
    except Exception as e:
        return f"[Playwright Error] {e}"

# WRAPPER FUNCTION TO CALL THE ABOVE
def extract_article_text(url):
    text = extract_with_requests(url)
    if text:
        return text
    else:
        try:
            return asyncio.run(extract_with_browser_async(url))
        except RuntimeError:
            return asyncio.get_event_loop().run_until_complete(extract_with_browser_async(url))
        

def keyword_occurrences(sentence: str, company_name: str) -> Dict[str, int]:
    keywords = [
        "cyberattack", "keylogger", "cyberattacks", "cyber", "subdomains", "driveby", "modifiable", "malicious",
        "malware", "countermeasures", "cybercrime", "phishing", "vishing", "weaponised", "multifactor", "backdoors",
        "opensource", "runtimes", "unencrypted", "hacktivists", "embed", "reconfigurations", "unpatched", "gapped",
        "firewalls", "timezones", "rootkits", "untrusted", "fuzzing", "configured", "wideband", "remotely", "flaw",
        "alerts", "eavesdropping", "cybersecurity", "validations", "simulating", "router", "hackers",
        "microcontroller authenticator", "intrusion", "cryptographically", "unsecure", "licenced", "directive",
        "spoofing", "sniffing", "overwriting", "encryption", "interface", "encrypted", "sanitization",
        "nontechnical", "exploiter", "disinformation", "ransomware"
    ]
    result = {}
    for word in [company_name] + keywords:
        # Use regex to match whole words, ignoring case
        pattern = rf'\b{re.escape(word)}\b'
        count = len(re.findall(pattern, sentence, flags=re.IGNORECASE))
        result[word] = count
    return result


def download_all(company, csv_name, n = 0, start = 0, skip = 0, devided = False):
    uploads_path = '/Users/bibbi/Desktop/tesi isa/tesi/data'
    subfolder_path = os.path.join(uploads_path, company)
    csv_path = os.path.join(subfolder_path, csv_name)
    df = pd.read_csv(csv_path)
    if devided:
        df = df.iloc[start: start+skip]
    text_void = []

    # EXTRACT FOR EACH LINK (only first 250)
    for link in tqdm(df["URL"]):
        try:
            article = extract_article_text(link)
            if 'Page not found' in article or ''==article or 'Access to this page has been denied' in article or 'Error' in article or 'Attention Required' in article or len(article)<500:
                article = extract_article_text(link)
            text_void.append(article)
        except Exception as e:
            text_void.append(f"[Fatal Error] {e}")

    # STORE website texts
    df["text_full"] = text_void
    # Apply the function to each row and expand the dictionary into columns
    df_keywords = df["text_full"].apply(lambda s: keyword_occurrences(s, company)).apply(pd.Series)
    df = pd.concat([df, df_keywords], axis=1)
    download_path = f"{uploads_path}_extracted/{company}"
    os.makedirs(download_path, exist_ok=True)
    if not devided:
        df.to_csv(f"{download_path}/extracted_{csv_name}", index=False, encoding = 'utf-8', errors = 'replace')
    else:
        df.to_csv(f"{download_path}/{n}_extracted_{csv_name}", index=False, encoding = 'utf-8', errors = 'replace')

    count = 0
    for i in range(len(df)):
        if 'Page not found' in df['text_full'].iloc[i] or ''==df['text_full'].iloc[i] or 'Access to this page has been denied' in df['text_full'].iloc[i] or 'Error' in df['text_full'].iloc[i] or 'Attention Required' in df['text_full'].iloc[i] or len(df['text_full'].iloc[i])<500:
            count += 1
    print(f'Number of errors for company {company}: {count}')

    return df


def estimate_tokens(text):
    return max(1, len(str(text)) // 4)


def calculate_splits(df, prompt, token_limit):
    with open(prompt, "r") as file:
        prompt = file.read()

    prompt_tokens = estimate_tokens(prompt)

    count_total = 0
    list_splits = []
    start = 0
    for i in range(0,len(df)):
        tokens = estimate_tokens(df['text_full'].iloc[i])
        if (count_total + tokens + prompt_tokens)>= token_limit:
            count_total = 0
            list_splits.append((start,i))
            start = i
            continue
        else:
            if i == (len(df)-1):
                list_splits.append((start,len(df)))
        count_total += tokens
    return list_splits


def prompt_2_call(list_text, company):
    with open("prompt_2.txt", "r") as file:
        prompt = file.read()

    prompt = prompt.replace('{company_name}', company)
    formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(list_text)])
    prompt = prompt.replace('{numbered_sequence_of_text_passages}', formatted_texts)
    prompt = prompt.replace('{input_lenght}', str(len(list_text)))

    msg = 'You are a cybersecurity analyst.'
    response = generate(msg, prompt)
    # print('------ responses ------', response)
    response_cleaned = response.strip().replace("json","").replace("```", "").replace("python\n", "").strip()
    cyberattack_flags = ast.literal_eval(str(response_cleaned))
    return cyberattack_flags


# Function to extract context snippet around the value
def extract_context(text):
    value_pattern = r'(\$[\d,.]+|€[\d,.]+|£[\d,.]+|\b\d+(?:\.\d+)?\s?(?:million|billion|thousand|k|%)\b|\bUSD\s?\d+|\d+\s?(?:percent|%))'
    if pd.isna(text):
        return None
    match = re.search(value_pattern, text, flags=re.IGNORECASE)
    if match:
        start, end = match.span()
        snippet_start = max(start - 30, 0)
        snippet_end = min(end + 30, len(text))
        return text[snippet_start:snippet_end]
    return None

def prompt_1_call(list_text):
    with open("prompt_1.txt", "r") as file:
        prompt = file.read()

    prompt = prompt.replace('{input_lenght}', str(len(list_text)))
    formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(list_text)])
    prompt = prompt.replace('{numbered_sequence_of_text_passages}', formatted_texts)
    
    msg = 'You are an expert in cybersecurity and economic value extraction.'
    response = generate(msg, prompt)
    print('------- response ---------',response)
    # response_cleaned = response.strip().replace("json", "").replace("```", "").replace("python\n", "").strip()
    # print(response_cleaned)
    # response = ast.literal_eval(str(response_cleaned))
    response_cleaned = response.strip()
    print(response_cleaned)
    m = re.search(r"```json(.*?)```", response_cleaned, re.DOTALL)
    response = ast.literal_eval(m.group(1))

    return response


# def prompt_3_call(list_text, list_values, list_descriptions, numbered_sequence_of_economic_values, company = 'Apple'):
#     with open("prompt_3.txt", "r") as file:
#         prompt = file.read()

#     prompt = prompt.replace('{company_name}', company)
#     formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(list_text)])
#     prompt = prompt.replace('{numbered_sequence_of_text_passages}', formatted_texts)
#     formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(list_values)])
#     prompt = prompt.replace('{numbered_sequence_of_economic_values}', str(numbered_sequence_of_economic_values))
#     formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(list_descriptions)])
#     prompt = prompt.replace('{numbered_sequence_of_descriptions_of_what_the_value_refer_to}', formatted_texts)

#     msg = 'You are an expert in cybersecurity and economic value extraction.'
#     response = generate(msg, prompt)
#     response_cleaned = response.strip()
#     print(response_cleaned)
#     m = re.search(r"```json(.*?)```", response_cleaned, re.DOTALL)
#     response = ast.literal_eval(m.group(1))

#     return response


def prompt_3_call(list_text, list_values, list_descriptions, numbered_sequence_of_economic_values, company="Apple"):
    # Load template
    with open("prompt_3.txt", "r") as file:
        prompt = file.read()

    # Replace placeholders
    prompt = prompt.replace('{company_name}', company)

    formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(list_text)])
    prompt = prompt.replace('{numbered_sequence_of_text_passages}', formatted_texts)

    formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(list_values)])
    prompt = prompt.replace('{numbered_sequence_of_economic_values}', str(numbered_sequence_of_economic_values))

    formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(list_descriptions)])
    prompt = prompt.replace('{numbered_sequence_of_descriptions_of_what_the_value_refer_to}', formatted_texts)

    # Message and model call
    msg = 'You are an expert in cybersecurity and economic value extraction.'
    response = generate(msg, prompt)

    # Clean response
    response_cleaned = response.strip()
    print(response_cleaned)  # Debugging: see what the model outputs

    # Extract fenced block (```json ... ```)
    m = re.search(r"```(?:json|python)?\s*(.*?)```", response_cleaned, re.DOTALL)
    payload = (m.group(1) if m else response_cleaned).strip()

    # Try JSON first, fallback to Python literal
    try:
        response = json.loads(payload)
    except json.JSONDecodeError:
        response = ast.literal_eval(payload)

    return response

def dict_cleaning(final_dict):
    cleaned_dict = {}
    for k, values in final_dict.items():
        # Rimuovi duplicati mantenendo l'ordine
        unique_values = []
        for v in values:
            if pd.notna(v):
                unique_values.append(v)
        cleaned_dict[k] = [unique_values]
    return cleaned_dict

def try_days_param(company, date):
    tot = 250
    days_param = 2
    while tot >= 250:
        tot = download_csv(company, date, days_param, try_ = True)
        if tot >= 250:
            days_param = days_param/2
            
    print(f'Correct paramenter : {days_param}')
    return days_param