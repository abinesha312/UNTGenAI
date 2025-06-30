    import urllib.parse
    import urllib3
    import json
    import aiohttp
    import asyncio
    import torch
    import os
    import time
    import re
    import signal
    from uuid import uuid4
    import requests
    from bs4 import BeautifulSoup, NavigableString
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from urllib.parse import urlparse, parse_qs

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    class Config:
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        SCRAPED_DATA_FILE = 'scraped_data.json'
        SCRAPED_URLS_FILE = 'scraped_urls.json'
        TIME_LOG_FILE = 'time_log.json'
        MAX_WEBSITES = 100000
        TIMEOUT = 10
        RETRIES = 3
        MAX_DEPTH = 10000
        MAX_URL_SCRAPE_TIME = 60
        UNT_PATTERN = re.compile(r'^https?://(?:[\w-]+\.)*unt\.edu(?:/[\w-]+)*(?:\?[\w=&]+)?')\

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    class WebScraper:
        '''
            This one the main part where it handles all the Data Processing part happens, Including validating the extracted text.
                - is_valid_response : Responsible for validating the responce extracted content.
                - fetch_page_content : Retrives the content from the website using async and creates session for for the whole scrapping
                - clean_text : It cleans the extracted text form the website when it have more spaces ih extracted code.
                - is_valid_content : Basically it validate whether we have some content or not after extraction.
                - is_complex_url : when the URL's are being stored in the content it check some URL's might havae Long URLs contains the session and ID to remov
    e that.
                - parse_content : Core Component where the extracted content are stored in the Structured JSON format for ease retrival and accessible.
                - extract_links :
        '''
        def __init__(self):
            self.scraped_urls = set()
            self.scraped_data = {}
            self.min_count = float('inf')
            self.max_count = 0
            self.min_count_url = ''
            self.max_count_url = ''
            self.total_count = 0
            self.document_count = 0
            self.processed_content = set()
            self.unscrapped_urls = set()
            self.unt_domains = [
                'https://www.unt.edu',
                'https://online.unt.edu',
                'https://library.unt.edu',
                'https://digital.library.unt.edu',
                'https://texashistory.unt.edu',
                'https://cybercemetery.unt.edu',
                'https://learn.unt.edu',
                'https://my.unt.edu',
                'https://studentaffairs.unt.edu',
                'https://registrar.unt.edu',
                'https://admissions.unt.edu',
                'https://financialaid.unt.edu',
                'https://housing.unt.edu',
                'https://dining.unt.edu',
                'https://it.unt.edu',
                'https://recsports.unt.edu',
                'https://business.unt.edu',
                'https://coe.unt.edu',
                'https://engineering.unt.edu',
                'https://music.unt.edu',
                'https://cvad.unt.edu',
                'https://cas.unt.edu',
                'https://hsc.unt.edu',
                'https://cob.unt.edu',
                'https://class.unt.edu',
                'https://math.unt.edu',
                'https://physics.unt.edu',
                'https://chemistry.unt.edu',
                'https://biology.unt.edu',
                'https://psychology.unt.edu',
                'https://history.unt.edu',
                'https://english.unt.edu',
                'https://philosophy.unt.edu',
                'https://economics.unt.edu',
                'https://geography.unt.edu',
                'https://sociology.unt.edu',
                'https://politicalscience.unt.edu',
                'https://linguistics.unt.edu',
                'https://worldlanguages.unt.edu',
                'https://communication.unt.edu',
                'https://journalism.unt.edu',
                'https://vpn.unt.edu',
                'https://sso.unt.edu',
                'https://eaglemail.unt.edu',
                'https://jobs.unt.edu',
                'https://careers.unt.edu',
                'https://transportation.unt.edu',
                'https://facilities.unt.edu',
                'https://ipa.unt.edu',
                'https://research.unt.edu',
                'https://ams.unt.edu',
                'https://its.unt.edu',
                'https://aits.unt.edu',
                'https://ecs.unt.edu',
                'https://guides.library.unt.edu',
                'https://discover.library.unt.edu',
                'https://findingaids.library.unt.edu',
                'https://journals.library.unt.edu',
                'https://esports.library.unt.edu',
                'https://untpress.unt.edu',
                'https://northtexan.unt.edu',
                'https://tams.unt.edu',
                'https://honors.unt.edu'
            ]

        def is_valid_response(self, status_code):
            return 200 <= status_code <= 205

        async def fetch_page_content(self, session, url, timeout=Config.TIMEOUT, retries=Config.RETRIES):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
            }
            attempt = 0
            while attempt < retries:
                try:
                    print(f"URL Before Encoding{url}")
                    # encoded_url = urllib.parse.quote(url, safe=':/')
                    encoded_url = url
                    print(f"Fetching URL: {encoded_url}")
                    async with session.get(encoded_url, headers=headers, timeout=timeout) as response:
                        if self.is_valid_response(response.status):
                            html_content = await response.text(errors='replace')
                            if "<html" not in html_content.lower():
                                print(f"Warning: The content fetched from {url} does not appear to be valid HTML.")
                                return None

                            if not self.is_valid_content(html_content):
                                print(f"Warning: The content fetched from {url} does not contain valid data.")
                                return None

                            return html_content
                        else:
                            print(f"Invalid response for {url}: {response.status}")
                            return None
                except asyncio.TimeoutError:
                    attempt += 1
                    print(f"Timeout while fetching {url}, attempt {attempt} of {retries}")
                    if attempt >= retries:
                        print(f"Skipping {url} after {retries} attempts due to timeout.")
                        return None
                    await asyncio.sleep(2)
                except aiohttp.ClientError as e:
                    print(f"Failed to scrape {url}: {e}")
                    return None

        def clean_text(self, text):
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\d+/\d+", "", text)
            return text.strip()

        def is_valid_content(self, text):
            """
                --  Check a valid URL or not
            """
            return len(self.clean_text(text)) >= 500

        def has_substantial_content(self, url):
            try:
                response = requests.get(url, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                word_count = len(text.split())
                paragraph_count = len(re.findall(r'\n\n', text))
                if word_count > 100 and paragraph_count > 2:
                    return True
                else:
                    return False
            except:
                return False

        def is_complex_url(self, url):
            """
                -- This one check the URL has more than 2 Quesry paramters in the URL also. the length after the quesry parameters
            """
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)

            if len(query_params) > 2 or any(len(v[0]) > 50 for v in query_params.values()):
                return True

            if re.search(r'(%3C|%3E|%3D|%26|%27|%22)', url):
                return True

            return False

        def remove_html_tags(self, text):
            return re.sub(r'<[^>]+>', '', text)

        def remove_css_js(self, text):
            text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text)
            text = re.sub(r'style\s*=\s*"[^"]*"', '', text)
            text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', text)
            text = re.sub(r'on\w+\s*=\s*"[^"]*"', '', text)
            return text

        def parse_content(self, html_content, url):
            soup = BeautifulSoup(html_content, 'lxml')
            for script in soup(["script", "style"]):
                script.decompose()
            body = soup.find('body')
            body_elements = []
            if body:
                for element in body.find_all(['p', 'div']):
                    extracted_text = self.extract_text_with_links(element, url)
                    if extracted_text:
                        body_elements.append(self.remove_html_tags(extracted_text))

            text_data = ' '.join(body_elements)
            text_data = self.remove_css_js(text_data)
            text_data = self.remove_html_tags(text_data)

            if len(text_data) < 150:
                return None

            images = soup.find_all('img', src=True)
            image_data = list(set([urllib.parse.urljoin(url, img['src']) for img in images]))
            headings = {f"h{i}": list(set([self.clean_text(h.get_text(strip=True)) for h in soup.find_all(f'h{i}')])) for i in range(1, 7)}
            lists = list(set([self.clean_text(ul.get_text(separator=", ", strip=True)) for ul in soup.find_all('ul')]))
            urls = list(set(self.extract_links(soup, url)))
            DataCount = len(text_data.split())

            structured_data = {
                "Source_URL": url,
                "Body": body_elements,
                "text": text_data,
                "images": image_data,
                "headings": headings,
                "lists": lists,
                "URLs": urls,
                "Count": DataCount
            }

            if DataCount < self.min_count:
                self.min_count = DataCount
                self.min_count_url = url
            if DataCount > self.max_count:
                self.max_count = DataCount
                self.max_count_url = url
            self.total_count += DataCount
            self.document_count += 1

            return structured_data

        def extract_text_with_links(self, element, base_url):
            result = []
            for content in element.contents:
                if isinstance(content, NavigableString):
                    text = self.clean_text(str(content))
                    if text and text not in self.processed_content:
                        result.append(text)
                        self.processed_content.add(text)
                elif content.name == 'a' and content.has_attr('href'):
                    link_text = self.clean_text(content.get_text())
                    link_url = urllib.parse.urljoin(base_url, content['href'])
                    if not self.is_complex_url(link_url) and Config.UNT_PATTERN.match(link_url):
                        link_format = f'({link_text}: {link_url})'
                        if link_format not in self.processed_content:
                            result.append(link_format)
                            self.processed_content.add(link_format)
                    elif link_text and link_text not in self.processed_content:
                        result.append(link_text)
                        self.processed_content.add(link_text)
                else:
                    result.append(self.extract_text_with_links(content, base_url))
            return ' '.join(filter(None, result))

        def extract_links(self, soup, base_url):
            links = []
            for a in soup.find_all('a', href=True):
                full_url = urllib.parse.urljoin(base_url, a['href'])
                if not self.is_complex_url(full_url) and Config.UNT_PATTERN.match(full_url):
                    links.append(full_url)
            return links

        def load_scraped_urls(self):
            try:
                with open(Config.SCRAPED_URLS_FILE, 'r', encoding='utf-8') as file:
                    urls_data = json.load(file)
                    self.scraped_urls = set(urls_data.values())
            except FileNotFoundError:
                self.scraped_urls = set()
            print(f"Loaded {len(self.scraped_urls)} previously scraped URLs.")

        def save_scraped_data(self):
            numbered_data = {f"URL_{str(i+1).zfill(2)}": data for i, data in enumerate(self.scraped_data.values())}
            with open(Config.SCRAPED_DATA_FILE, 'w', encoding='utf-8') as file:
                json.dump(numbered_data, file, indent=4, ensure_ascii=False)

        def save_unscrapped_urls(self):
            unscrapped_data = {f"URL_{str(i+1).zfill(2)}": url for i, url in enumerate(self.unscrapped_urls)}
            with open('unscrapped_urls.json', 'w', encoding='utf-8') as file:
                json.dump(unscrapped_data, file, indent=4, ensure_ascii=False)
            print(f"Saved {len(self.unscrapped_urls)} unscrapped URLs to unscrapped_urls.json")


        def save_scraped_urls(self):
            urls_data = {f"URL_{str(i+1).zfill(2)}": url for i, url in enumerate(self.scraped_urls)}
            with open(Config.SCRAPED_URLS_FILE, 'w', encoding='utf-8') as file:
                json.dump(urls_data, file, indent=4, ensure_ascii=False)

        async def scrape_domain_and_related(self, session, start_url, depth=0):
            if start_url in self.scraped_urls or depth > Config.MAX_DEPTH or len(self.scraped_urls) >= Config.MAX_WEBSITES:
                return {}

            print(f"Scraping: {start_url} (Depth: {depth})")
            try:
                html_content = await self.fetch_page_content(session, start_url)
                if html_content is None:
                    print(f"Skipping {start_url} due to fetch failure")
                    self.unscrapped_urls.add(start_url)
                    return {}

                structured_data = self.parse_content(html_content, start_url)
                self.scraped_urls.add(start_url)
                self.scraped_data[start_url] = structured_data
                data = {start_url: structured_data}

                if depth < Config.MAX_DEPTH:
                    related_urls = self.extract_links(BeautifulSoup(html_content, 'lxml'), start_url)
                    tasks = []
                    for url in related_urls:
                        if url not in self.scraped_urls and len(self.scraped_urls) < Config.MAX_WEBSITES:
                            task = self.scrape_domain_and_related(session, url, depth + 1)
                            tasks.append(task)
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for result in results:
                            if isinstance(result, dict):
                                data.update(result)
                            elif isinstance(result, Exception):
                                print(f"Error occurred while scraping a related URL: {result}")

                return data
            except Exception as e:
                print(f"Error occurred while scraping {start_url}: {e}")
                self.unscrapped_urls.add(start_url)
                return {}


        async def scrape_all_domains(self):
            async with aiohttp.ClientSession() as session:
                for domain in self.unt_domains:
                    print(f"Starting to scrape domain: {domain}")
                    domain_data = await self.scrape_domain_and_related(session, domain)
                    self.scraped_data.update(domain_data)
                    self.save_scraped_data()
                    self.save_scraped_urls()
                    self.save_unscrapped_urls()
                    print(f"Completed scraping domain: {domain}")
                    print(f"Total URLs scraped: {len(self.scraped_urls)}")
                    if len(self.scraped_urls) >= Config.MAX_WEBSITES:
                        print(f"Reached maximum number of websites ({Config.MAX_WEBSITES}). Stopping.")
                        break
                    await asyncio.sleep(5)

    class VectorStoreBuilder:
        '''
        This one code is to Store the data into the Vector DB and Build them
            Following functions
            - Create_document(scraped_data) : Creates the documents where indivudal document is stored
            - build_vector_store(documents) : Builds the Vectore Store with the created Doucments which are being chucked.
        '''
        def create_documents(self, scraped_data, output_file='documents.json'):
            documents = []
            for url, data in scraped_data.items():
                if data is None:
                    print(f"Warning: No data for URL {url}")
                    continue
                if not isinstance(data, dict):
                    continue
                if 'text' not in data or not data['text']:
                    continue

                doc = Document(
                    page_content=data['text'],
                    metadata={
                        "source": data.get('Source_URL', url),
                        "word_count": data.get('Count', 0),
                        "original_id": str(uuid4())
                    }
                )
                documents.append(doc)

            # Convert documents to a serializable format
            serializable_docs = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in documents
            ]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_docs, f, ensure_ascii=False, indent=4)

            print(f"Created and stored {len(documents)} documents in {output_file}")
            return documents

        @staticmethod
        def build_vector_store(documents):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': device}
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
            )

            all_splits = []
            for doc in documents:
                splits = text_splitter.split_documents([doc])
                for i, split in enumerate(splits):
                    split.metadata["chunk_id"] = f"{doc.metadata['original_id']}_chunk_{i+1}"
                all_splits.extend(splits)

            db = FAISS.from_documents(all_splits, embeddings)
            db.save_local(Config.DB_FAISS_PATH)

            print(f"Total documents after splitting: {len(all_splits)}")
            return all_splits

    class TimeLogger:
        '''
        This code is to maintain the log the details from After running the script.
            Following Funciton.
                - log_time(start_time, end_time, scraper) : Log the start time, end time, responce time, document statistics which is document dount min, max, A
    vg
        '''
        @staticmethod
        def log_time(start_time, end_time, scraper):
            response_time = end_time - start_time
            avg_count = scraper.total_count / scraper.document_count if scraper.document_count > 0 else 0

            time_log = {
                "start_time": time.ctime(start_time),
                "end_time": time.ctime(end_time),
                "response_time": f"{response_time:.6f} seconds",
                "document_statistics": {
                    "min_count": {
                        "count": scraper.min_count,
                        "url": scraper.min_count_url
                    },
                    "max_count": {
                        "count": scraper.max_count,
                        "url": scraper.max_count_url
                    },
                    "average_count": f"{avg_count:.2f}",
                    "total_documents": scraper.document_count
                }
            }

            with open(Config.TIME_LOG_FILE, 'w', encoding='utf-8') as file:
                json.dump(time_log, file, indent=4, ensure_ascii=False)

            print(f"Response time: {response_time:.6f} seconds")
            print(f"Minimum document count: {scraper.min_count} (URL: {scraper.min_count_url})")
            print(f"Maximum document count: {scraper.max_count} (URL: {scraper.max_count_url})")
            print(f"Average document count: {avg_count:.2f}")
            print(f"Total documents: {scraper.document_count}")

    class VectorDBCreator:
        '''
        Initial function where it starts the function all the process start here.
            - Singal Handler : responsible for handling the function where the user data input to start and stop command are monitored
        '''
        def __init__(self):
            self.scraper = WebScraper()
            self.vector_store_builder = VectorStoreBuilder()
            self.time_logger = TimeLogger()

        def signal_handler(self, signum, frame):
            print("Received termination signal. Saving data and exiting...")
            self.scraper.save_scraped_data()
            self.scraper.save_scraped_urls()
            documents = self.vector_store_builder.create_documents(self.scraper.scraped_data)
            if documents:
                self.vector_store_builder.build_vector_store(documents)
            exit(0)

        async def create_vector_db(self):
            start_time = time.time()
            try:
                self.scraper.load_scraped_urls()
                await asyncio.wait_for(self.scraper.scrape_all_domains(), timeout=25200)

                if self.scraper.scraped_data:
                    documents = self.vector_store_builder.create_documents(self.scraper.scraped_data)
                    if documents:
                        self.vector_store_builder.build_vector_store(documents)
                    else:
                        print("No valid documents to build vector store.")
                else:
                    print("No scraped data available.")

                documents = self.vector_store_builder.create_documents(self.scraper.scraped_data)
                if documents:
                    self.vector_store_builder.build_vector_store(documents)
                else:
                    print("No valid documents to build vector store.")
            except asyncio.TimeoutError:
                print("Scraping process timed out after 7 hours.")
            except Exception as e:
                print(f"An error occurred: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
            finally:
                end_time = time.time()
                self.time_logger.log_time(start_time, end_time, self.scraper)

                print(f"Total URLs scraped: {len(self.scraper.scraped_urls)}")
                print(f"Total documents processed: {self.scraper.document_count}")
                print(f"Scraped data entries: {len(self.scraper.scraped_data)}")

    if __name__ == "__main__":
        vector_db_creator = VectorDBCreator()
        signal.signal(signal.SIGINT, vector_db_creator.signal_handler)
        signal.signal(signal.SIGTERM, vector_db_creator.signal_handler)
        asyncio.run(vector_db_creator.create_vector_db())