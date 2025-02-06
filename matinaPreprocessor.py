import emoji
import unicodedata
import regex as re
from collections import Counter
from nltk.corpus import stopwords


from piraye import NormalizerBuilder
from piraye.normalizer_builder import Config


# Define Persian Unicode ranges and other necessary constants
persian_unicode_range = re.compile(r'[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]')
non_persian_unicode_range = re.compile(r'[^\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF\s\W]')
symbol_regex = re.compile(r'/(?!/w).+? s]|#d')#r'[^\w\s]'
numeric_regex = re.compile(r'\d')
eng_threshold = 0.5
class preprocess_docs():
    def __init__(self, doc_origin, #doc_text,
                 english_allowed=False, arabic_allowed=False,
                 doc_lang_thresh=0.5, doc_num_thresh=0.8, doc_symb_thresh=0.8, 
                 doc_word_length=[3,10], doc_stopword_thresh=0.1, short_doc_thresh=150,
                 shortLine_proportion_thresh=0.8, english_lines=True,
                 cons_new_lines=True, cons_chars=True, non_sense_patterns=True, 
                 numeric_lines=True, symbolic_lines=True, personal_info=True, emojis=False,
                 cons_newline_thresh=2, cons_chars_thresh=3, numeric_lines_thresh=0.8,
                 symbolic_lines_thresh=0.8, repeated_lines_thresh=10,
                 norm_dates=True, norm_numbers=True, norm_symbols=True
                ):
        
        self.emoji_chars_codes = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons -> emojies in multilingual text
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u'\U00010000-\U0010ffff'
                           # u"\u200d"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\u3030"
                           u"\ufe0f"
                           u"\u2069"
                           u"\u2066"
                           # u"\u200c"
                           u"\u2068"
                           u"\u2067"
                           #u"\u0640" arabic tatweel "-"
                           "]+", flags=re.UNICODE)
        
        self.symbol_pattern = re.compile(  r'[\U00010000-\U0010FFFF]'  # Characters beyond the BMP
                                            r'|[\ue000-\uf8ff]'          # Supplementary Private Use Area-A
                                            r'|[\u0b80-\u0bff]'          # Tamil
                                            r'|[\ufe70-\ufeff]'          # Arabic Presentation Forms-B
                                            r'|[\uF000-\uF0FF]',         # Additional private use characters or special symbols
                                            re.UNICODE
                                        )
        self.source_web_pattern = re.compile(r'^(https?://\S+|www\.\S+|t.me/\S+|tel.me/\S+|\n\S+\.ir/\S+|\n\S+\.com/|S+)(?:\n|\.|$)', re.MULTILINE)
        
        self.id_patterns = r'(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([A-Za-z]*(?:[\p{Pd}]*\p{Nd}){6,})(?:$|[\b\s@?,!;:\'\")(.\p{Han}])'
        self.key_pattern = r'(?:^|[\b\s@?,!:;\'\")(.\p{Han}])((?:(?:[A-Za-z]+[\p{Nd}\p{Pd}\/\+\=:_]+|[\p{Nd}\p{Pd}\/\+\=:]+[A-Za-z]+)){4,}|(?:(?:\p{Nd}{3,}|[A-Z]+\p{Nd}+[A-Z]*|\p{Nd}+[A-Z]+\p{Nd}*)[ \p{Pd}]?){3,})(?:$|[\b\s\p{Han}@?,!;:\'\")(.])'
        self.ipv4_pattern = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}'
        self.ipv6_pattern = r'(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
        self.ip_pattern = r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])(" + r"|".join([self.ipv4_pattern, self.ipv6_pattern]) + ")(?:$|[\s@,?!;:\'\"(.\p{Han}])"
        self.email_pattern = re.compile(r"""
                                        (?:[\s_])*
                                        [\w\.\+]+                  # Username part: can include letters, numbers, dots, underscores, and pluses                # Optional spaces or underscores within the username
                                        @                          # At symbol
                                        (?:[\w\.\+\s]+\.)+         # Domain part: can include letters, numbers, dots, pluses, and spaces
                                        \w{2,}                     # Top-level domain: at least two word characters
                                        """, 
                                        re.VERBOSE)
        self.shaba_pattern =  r'IR\d{24}|IR\d{2}-\d{4}-\d{4}-\d{4}- \d{4}'
        self.account_number_pattern = r'\d{4}-\d{4}-\d{4}-\d{4}'
        
        self.doc_origin  = doc_origin 
        # self.docs = doc_text

        self.normalizer = NormalizerBuilder([Config.PUNCTUATION_FA, Config.ALPHABET_FA, Config.DIGIT_FA],tokenization=True).build()
        self.short_doc_thresh = short_doc_thresh
        self.english_allowed = english_allowed
        self.arabic_allowed = arabic_allowed
        self.doc_lang_thresh = doc_lang_thresh
        self.doc_num_thresh = doc_num_thresh
        self.doc_symb_thresh = doc_symb_thresh
        self.doc_word_length = doc_word_length
        self.doc_stopword_thresh = doc_stopword_thresh
        self.short_line_thresh = 10
        self.shortLine_proportion_thresh = shortLine_proportion_thresh
        
        
        self.cons_new_lines = cons_new_lines
        self.cons_chars = cons_chars
        self.non_sense_patterns = non_sense_patterns
        self.numeric_lines = numeric_lines
        self.symbolic_lines = symbolic_lines
        self.english_lines = english_lines
        self.personal_info = personal_info
        self.emojis = emojis
        self.cons_newline_thresh = cons_newline_thresh
        self.cons_chars_thresh = cons_chars_thresh
        self.numeric_lines_thresh = numeric_lines_thresh
        self.symbolic_lines_thresh = symbolic_lines_thresh
        self.repeated_lines_thresh = repeated_lines_thresh
        
        self.norm_dates = norm_dates
        self.norm_numbers = norm_numbers
        self.norm_symbols = norm_symbols

        self.persian_stopwords = self.load_stopwords()
        self.arabic_stopwords = self.load_stopwords(False)
        self.english_stopwords = list(stopwords.words('english'))


    def load_stopwords(self, persian=True):
        if persian:
            with open('/mnt/old/home/taheri/dataCleaning/persianStopwords.txt', 'r') as f:
                stopwords = f.read()
                return stopwords.split('\n')
        else:
            with open('/mnt/old/home/taheri/dataCleaning/dialected_arabic_stopwords.txt', 'r') as f:
                stopwords = f.read()
                return stopwords.split('\n')
            
            
    def eliminate_if_short(self, doc):
        if len(doc.split()) < self.short_doc_thresh:
            # print("shortttt")
            # print(doc)
            return 
        else: 
            return doc
        
    def is_persian_char(self, char):
        return persian_unicode_range.match(char)
    
    def is_non_persian_char(self, char):
        return non_persian_unicode_range.match(char)
    
    def calculate_paragraphs(self, text):
        # Split the text into paragraphs by newline
        paragraphs = text.split('\n')
        # Count the total number of paragraphs
        total_paragraphs = len(paragraphs)
        # Count the number of paragraphs with fewer than 15 words
        short_paragraphs = sum(len(paragraph.split()) < 15 for paragraph in paragraphs)
        # Return both counts
        return total_paragraphs, short_paragraphs
    
    def eliminate_document_level(self, doc):
        if not self.eliminate_if_short(doc):
            # print("short")
            return 
        # 0. check the number of hashtags
        hashtag_count = doc.count('#')
        if hashtag_count > 10:
            return
        
        if 'جزییات پخش' in doc:
            return
        
        total_paragraphs, short_paragraphs = self.calculate_paragraphs(doc)
        if short_paragraphs/total_paragraphs > 0.4:
            return
        
        # 1. Check if more than 50% non-Persian text
        persian_chars = sum(1 for char in doc if self.is_persian_char(char))
        non_persian_chars = sum(1 for char in doc if self.is_non_persian_char(char))
        total_chars = len(doc)
        
        # Count the number of English characters
        english_characters = sum(1 for char in doc if char.isalpha() and char.isascii())
        total_characters = len(doc)
        if total_characters == 0:
            return
        ratio = english_characters / total_characters
        if ratio > eng_threshold:
            return

        if (non_persian_chars/total_chars > self.doc_lang_thresh and not self.english_allowed and not self.arabic_allowed):
            # print("persian ratio")
            return
        
        # 2. Calculate average word length
        words = re.findall(r'\b\w+\b', doc)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        if not (self.doc_word_length[0] <= avg_word_length <= self.doc_word_length[1]):
            # print("word_length")
            return
            
        # 3. Check symbol to word ratio
        symbols = symbol_regex.findall(doc)
        symbol_to_word_ratio = len(symbols) / len(words) if words else 0
        
        if symbol_to_word_ratio > self.doc_symb_thresh:
            # print("symbol")
            return 
            
        # 4. Check numeric characters ratio
        numerics = numeric_regex.findall(doc)
        numeric_to_word_ratio = len(numerics) / len(words) if words else 0
        
        if numeric_to_word_ratio > self.doc_num_thresh:
            # print("numeric")
            return
        
        # 5. Check for documents with short lines
        lines = doc.split('\n')
        short_lines = [line for line in lines if len(line.split(' ')) < self.short_line_thresh]
        if len(short_lines) / len(lines) >= self.shortLine_proportion_thresh:
            # print('short lines')
            return
        # 6. Check for minimum percentage of Persian stop words
        norm_words = re.findall(r'\b\w+\b', self.normalizer.normalize(doc))
        word_counter = Counter(norm_words)
        stop_word_count = sum(word_counter[word] for word in self.persian_stopwords if word in word_counter)
        stop_word_ratio = stop_word_count / len(words) if words else 0
        
        if (stop_word_ratio < self.doc_stopword_thresh and not self.english_allowed and not self.arabic_allowedh):
            # print("persian only")
            return
        elif (self.arabic_allowed or self.english_allowed):
            all_words = re.findall(r'\b\w+\b',doc)
            word_counter = Counter(all_words)
            arabic_stop_word_count = sum(word_counter[word] for word in self.arabic_stopwords if word in word_counter)
            arabic_stop_word_ratio = arabic_stop_word_count / len(words) if words else 0
            if (arabic_stop_word_ratio < self.doc_stopword_thresh and not self.english_allowed):
                # print("not arabic not persian")
                return
            
        return doc
    
        
    def preprocess_character_level(self, text):

        if self.norm_dates:
            text = self.normalize_dates(text)
            
        if self.norm_numbers:
            text = self.normalize_numbers(text)
            
        if self.norm_symbols:
            text = self.normalize_symbols(text)

        if self.cons_chars:
            text = self.remove_cons_chars(text)

        if self.non_sense_patterns:
            text = self.remove_non_sense_patterns(text)

        if self.emojis:
            text = self.remove_emojis(text)

        if self.personal_info:
            text = self.remove_personal_info(text)
            
        processed_lines = []
        for line in text.split('\n'):
            if len(line) > 2:
                if self.numeric_lines:
                    line = self.remove_numeric_lines(line)
                    
                if self.symbolic_lines and line:
                    # print("symbolic line")
                    line = self.remove_symbolic_lines(line)
                
                if self.english_lines and line:
                    # print("english_lines line")
                    line = self.remove_english_symbolic_lines(line)
                    
                if line:
                    # print("last line")
                    processed_lines.append(line)
                    
            elif len(line)==0:
                processed_lines.append('\n')
            text = '\n'.join(processed_lines)
            
        if self.cons_new_lines:
            text = self.remove_cons_new_lines(text)

        return text
            

    ''' remove consecutive new lines
        once befor and once after other preprocessings'''
    def remove_cons_new_lines(self, text):     
        # text = re.sub(r'\n+', '\n', text)
        # text = re.sub(r'\n+ | \n \n', '\n', text)
        # text = re.sub(r'\n+', '\n', text)
        # text = re.sub(r'\n\s*\t*\n*', '\n', text)
        
        ''' pattern detects newlines repeated more than thereshold 
            and replaces them with two new lines.'''
        pattern = r'(\n[\t ]*){' + str(self.cons_newline_thresh) + ',}'
        text = re.sub(pattern, '\n\n', text)
        return text

    '''replace consecutive repeated characters by two characters'''
    def remove_cons_chars(self, text):
        '''pattern detects consecutive repeated characters that are repeated
            more than the thereshold'''
        pattern = r'(.)\1{' + str(self.cons_chars_thresh - 1) + ',}'
        text = re.sub(pattern, r'\1\1', text)
        return text


    '''remove repeated lines (e.g. book names) from documents'''
    def remove_repeated_lines(self, text):      
        lines = text.split('\n')
        stripped_lines = [re.sub(r'^\d+', '', line).strip() for line in lines]
        line_counts = {line: stripped_lines.count(line) for line in set(stripped_lines)}
        
        filtered_lines = [re.sub(r'^\d+', '', line) for line, stripped_line in zip(lines, stripped_lines) if \
                                    line_counts.get(stripped_line, 0) < self.repeated_lines_thresh]
        return '\n'.join(filtered_lines)
            
    def clean_cultura(self, text):
        # Patterns for removal at the beginning
        patterns_start = [
            r'^[\w\s]+(-\s?\n?)?$',
            r'^به گزارش [^،؛]*[،؛]',  # Matches 'به گزارش ... ،' or 'به گزارش ... ؛'
            r'^- [^،؛]*[،؛]',         # Matches '- ... ،' or '- ... ؛'
            r'^اعتمادآنلاین\|',        # Matches 'اعتمادآنلاین|'
            r'افتاب‌نیوز : ',
            r'خبرورزشی -',
            r'^به گزارش [^،؛]*[،؛].*به گزارش [^،؛]*[،؛]',      # Matches repeated 'به گزارش' pattern at the start
            r'^به نقل [^،؛]*[،؛].*به گزارش [^،؛]*[،؛]',      # Matches repeated 'به گزارش' pattern at the start
            r'^به گزارش [^،؛]*[،؛].*به نقل [^،؛]*[،؛]',      # Matches repeated 'به گزارش' pattern at the start
            r'^\+ نوشته شده در .* توسط .*\|',  # Matches '+ نوشته شده در ... توسط ... |'
            r' [ ult_countdown count_style="ult - cd - s2″ datetime="2019/08/30 19:59:59″ countdown_opts="sday , shr , smin , ssec " br_style="solid " br_size="5″ br_color="#c97060″ timer_bg_color="#ca432c " br_time_space="40″ string_days="روز " string_days2="روز " string_hours="ساعت " string_hours2="ساعت " string_minutes="دقیقه " string_minutes2="دقیقه " string_seconds="ثانیه " string_seconds2="ثانیه " tick_col="#ff " tick_style="font - weight : bold ; " tick_size="desktop:80px ; " tick_line_height="desktop:100px ; " tick_unit_style="font - weight : bold ; " tick_sep_size="desktop:22px ; " tick_sep_line_height="desktop:30px ; " ] '
        ]
        
        # Remove matching patterns at the start
        for pattern in patterns_start:
            text = re.sub(pattern, '', text)
        
        # Patterns to remove specific lines or truncate from a point
        patterns_lines = [
            r'^READ.*',  # Matches lines starting with READ,
            r'تاریخ انتشار خبر : ',
            r'پیام \d+ پست \d+ دیدگاه',
            r'تاریخ به‌روزشدن سایت : ',
            r'کل بازدید ها : ',
            r'بازدیدهای این ماه : ',
            r'بازدید دیروز : ',
            r'بازدیدهای امروز : ',
            r'تعداد بازدید : \d+ تاریخ ایجاد : ',
            
            r'.*\( \+ ویدیو\)',  # Matches lines containing ( + ویدیو)
            r'.*سوابق : \d+ دیدگاه , \d{4} - \d{2} - \d{2} در \d{2} : \d{2} \S+ \. \S+.*',  # Matches "سوابق : (some number دیدگاه ) , YYYY - MM - DD در HH : MM ..."
            r'.*\d{4} - \d{2} - \d{2} در \d{2} : \d{2} ق . ظ.*',  # Matches dates like '۱۳۹۵ - ۰۴ - ۲۹ در ۶ : ۲۸ ق . ظ'
            r'.*\|.*\|.*\|.*\|.*',  # Matches lines that contain more than 3 "|"
            r'.*کد خبر : .*',  # Matches lines that contain 'کد خبر :'
            r'.*تاریخ انتشار : .*',  # Matches lines that contain 'تاریخ انتشار :'
            r'.*Next Article.*',  # Matches lines that contain 'Next Article'
            r'.*Previous Article.*',  # Matches lines that contain 'Previous Article'
            r'.*به روز شده در : .*',  # Matches lines that contain 'به روز شده در :'
            r'.*\d{2} : \d{2} - \d{4}/\d{2}/\d{2} / شماره : \d+ / تعداد نمایش : \d+',  # Matches lines like '۱۳ : ۱۹ - 1396/11/17 / شماره : ۴۹۹۲۰۲ / تعداد نمایش : ۹۷'
            r'.*, دنیای اقتصاد \d{2}/\d{1}/\d{2}',  # Matches lines like '، دنیای اقتصاد 17/4/98'
            r'.*\d{1}نفر \d{4}/\d{2}/\d{2} \d+ بازدید اشتراک گذاری',  # Matches lines like '1نفر 1400/11/20 82 بازدید اشتراک گذاری'
            r'.*کد : \d{17} .* \d{2}:\d{2}',  # Matches lines like 'کد : 13960419253997316 دوشنبه 19 تیر 96 ساعت 14:28'
        ]
        
        # Remove matching lines
        lines = text.splitlines()
        cleaned_lines = [line for line in lines if not any(re.search(pattern, line) for pattern in patterns_lines)]
        text = "\n".join(cleaned_lines)

        # Patterns for removal at the end
        patterns_end = [
            r'(پایان پیام|انتهای پیام)( منبع ?[:：] ?(\S+\s?){1,4})?$',  # Matches 'پایان پیام' or 'انتهای پیام' optionally followed by 'منبع'
            r'منبع ?[:：] ?(\S+\s?){1,4}$',                          # Matches 'منبع : ...' or 'منبع: ...' with at most 4 words
            r'\./ (\w+\s?){1,4}$'
        ]
        
        # Remove matching patterns at the end
        for pattern in patterns_end:
            text = re.sub(pattern, '', text)
        
        return text.strip()
      
    '''patterns that do not make sense and are seen in the dataset are removed'''
    def remove_non_sense_patterns(self, text):
        if self.doc_origin == 'telegram':
            pattern1 = 'Cover|@page {padding: 0pt; margin:0pt}|body { text-align: center; padding:0pt; margin: 0pt; }'
            pattern2 = 'Unknown'
            pattern3 = 'FIDIBO'
            combined_pattern = f"({pattern1}|{pattern2}|{pattern3})"
            text = re.sub(combined_pattern, '', text)

        if self.doc_origin == 'ghdoc':
            pattern1 = re.compile(r'.*مرکز تحقیقات رایانهاي قائمیه اصفهان.*', re.UNICODE)
            # Split the text into lines
            lines = text.split('\n')
            # Filter out lines that match the pattern
            filtered_lines = [line for line in lines if not pattern1.match(line)]
            text = '\n'.join(filtered_lines)
            
            pattern2 = r'ص \p{Nd}+\.\n|صفحه\p{Nd}+|صفحه \p{Nd}+ از \p{Nd}+|.*ص: \p{Nd}+.*|ص\p{Nd}+\n|ص:\p{Nd}+\n|\[\s*صفحه\s*\p{Nd}+\s*\]|\[\p{Nd}+\s+\[\s*صفحه\s*'
            pattern3 = 'Your browser does not support the audio tag.'
            pattern4 = 'Cover|@page {padding: 0pt; margin:0pt}|body { text-align: center; padding:0pt; margin: 0pt; }'
            pattern5 = 'Unknown'
            combined_pattern = f"({pattern2}|{pattern3}|{pattern4}|{pattern5})"
            text = re.sub(combined_pattern, '\n', text)
            
        if self.doc_origin == "chap_sch.ir_docs":
            pattern1 = r'ص \p{Nd}+\.\n|صفحه\p{Nd}+|صفحه \p{Nd}+ از \p{Nd}+|.*ص: \p{Nd}+.*|ص\p{Nd}+\n|ص:\p{Nd}+\n|\[\s*صفحه\s*\p{Nd}+\s*\]|\[\p{Nd}+\s+\[\s*صفحه\s*'
            combined_pattern = f"({pattern1})"
            text = re.sub(combined_pattern, '\n', text)
            
        if self.doc_origin == "takbook_1403_03_28":
            pattern1 = r'ص \p{Nd}+\.\n|صفحه\p{Nd}+|صفحه \p{Nd}+ از \p{Nd}+|.*ص: \p{Nd}+.*|ص\p{Nd}+\n|ص:\p{Nd}+\n|\[\s*صفحه\s*\p{Nd}+\s*\]|\[\p{Nd}+\s+\[\s*صفحه\s*'
            pattern2 = r'\bhttps?://\S+|www\.\S+\b'
            pattern3 = r'\bpg\.\s*\d+\b'
            pattern4 = r'\bص:\s*\d+\b'
            pattern5 = r'آکادمی مجازی باور مثبت www\.pba1\.com',
            pattern6 = r'Attributions:\nImages Credits: pch\.vector / Freepik',
            pattern7 = r'www\.khanehdastan\.ir بانک داستان های ترجمه چوک www\.chouk\.ir\n\d+'
            pattern8 = r'\U00100725\uf0fc'
            
            combined_pattern = f"({pattern1}|{pattern2}|{pattern3}|{pattern4}|{pattern5}|{pattern6}|{pattern7})"
            text = re.sub(combined_pattern, '\n', text)
            
            text = self.remove_repeated_lines(text)
            
            # Remove lines with only a number
            number_only_pattern = r'^\s*\d+\s*$'  # Lines with only a number
            text = re.sub(number_only_pattern, '', text, flags=re.MULTILINE)
        
        if (self.doc_origin == "papersPruned" or self.doc_origin == "UTpapers" or self.doc_origin == "irandoc"):
            pattern1 = r'[pic]'
            combined_pattern = f"({pattern1})"
            text = re.sub(combined_pattern, '', text)
        
        if self.doc_origin == "jahadPapers":
            pattern1 = r'[pic]'
            pattern2 = r'Archive of SID'
            pattern3 = r'www. SID. ir'
            pattern4 = r'SID'

            combined_pattern = f"({pattern1}|{pattern2}|{pattern3}|{pattern4})"
            text = re.sub(combined_pattern, '', text)    
        
        if (self.doc_origin == "baznashr" or self.doc_origin == "virgol"):
            pattern1 = r'document.createElement \( » video » \) ؛'
            pattern2 = r'https://deltapayamvideo.arvanvodmp4'
            pattern3 = r'<br/>'
            pattern4 = r' دانلود  فیلم اصلی'
            pattern5 = r'کد ویدیو'
            pattern6 = r'به این نوشته امتیاز بدهید'
            pattern7 = r'به\sاین\sنوشته\sامتیاز\sبدهید\s\[ مجموع : ۰ میانگین : ۰ \]'
            pattern8 = r'\n\d+xx  \( نمایش کامل \)' 
            
            combined_pattern = f"({pattern1}|{pattern2}|{pattern3}|{pattern4}|{pattern5}|{pattern6}|{pattern7}|{pattern8})"
            text = re.sub(combined_pattern, '', text)
            # pattern = re.compile(r'{ padding:0px ؛ margin : 0 ؛ padding - top:1em!important ؛ padding - bottom:1em!important ؛ width:100 % ؛ display : block ؛ font - weight : bold ؛ background - color : # ECF0F1 ؛ border:0!important ؛ border - left:4px solid # 00!important ؛ box - shadow : 0 1px 2px rgba ( 0 ، 0 ، 0 ، 0.17 ) ؛ -moz - box - shadow : 0 1px 2px rgba ( 0 ، 0 ، 0 ، 0.17 ) ؛ -o - box - shadow : 0 1px 2px rgba ( 0 ، 0 ، 0 ، 0.17 ) ؛ -webkit - box - shadow : 0 1px 2px rgba ( 0 ، 0 ، 0 ، 0.17 ) ؛ text - decoration : none ؛ } : active ، : hover { opacity : 1 ؛ transition : opacity 250ms ؛ webkit - transition : opacity 250ms ؛ text - decoration : none ؛ } { transition : background - color 250ms ؛ webkit - transition : background - color 250ms ؛ opacity : 1 ؛ transition : opacity 250ms ؛ webkit - transition : opacity 250ms ؛ } .ctaText { font - weight : bold ؛ color : # 3498DB ؛ text - decoration : none ؛ font - size : 16px ؛ } .postTitle { color : # 34495E ؛ text - decoration : underline!important ؛ font - size : 16px ؛ } : hover .postTitle { text - decoration : underline!important ؛ } ', re.DOTALL)
            pattern = re.compile(r'\.related - post h3 \{.*?\}\s*\.yarpp - related \.minientry \{.*?\}\s*\.yarpp - related \.post - info \{.*?\}\s*\.yarpp - related a \{.*?\}\s*', re.DOTALL)
            text = re.sub(pattern, ' ' , text)
        
        if (self.doc_origin == 'resalat'):
            pattern1 = r'\bص:\s*\d+\b|\bص :\s*\d+\b|\bصفحه\s*\d+\b'
            text = re.sub(pattern1, ' ' , text)
        
        if self.doc_origin == 'cultura':
            pattern1 = r'document.createElement \( » video » \) ؛'
            pattern2 = r'https://deltapayamvideo.arvanvodmp4'
            pattern3 = r'<br/>'
            pattern4 = r' دانلود  فیلم اصلی'
            pattern5 = r'کد ویدیو'
            pattern6 = r'به این نوشته امتیاز بدهید'
            pattern7 = r'به\sاین\sنوشته\sامتیاز\sبدهید\s\[ مجموع : ۰ میانگین : ۰ \]'
            pattern8 = r'\n\d+xx  \( نمایش کامل \)' 
            
            combined_pattern = f"({pattern1}|{pattern2}|{pattern3}|{pattern4}|{pattern5}|{pattern6}|{pattern7}|{pattern8})"
            text = re.sub(combined_pattern, '', text)
            # pattern = re.compile(r'{ padding:0px ؛ margin : 0 ؛ padding - top:1em!important ؛ padding - bottom:1em!important ؛ width:100 % ؛ display : block ؛ font - weight : bold ؛ background - color : # ECF0F1 ؛ border:0!important ؛ border - left:4px solid # 00!important ؛ box - shadow : 0 1px 2px rgba ( 0 ، 0 ، 0 ، 0.17 ) ؛ -moz - box - shadow : 0 1px 2px rgba ( 0 ، 0 ، 0 ، 0.17 ) ؛ -o - box - shadow : 0 1px 2px rgba ( 0 ، 0 ، 0 ، 0.17 ) ؛ -webkit - box - shadow : 0 1px 2px rgba ( 0 ، 0 ، 0 ، 0.17 ) ؛ text - decoration : none ؛ } : active ، : hover { opacity : 1 ؛ transition : opacity 250ms ؛ webkit - transition : opacity 250ms ؛ text - decoration : none ؛ } { transition : background - color 250ms ؛ webkit - transition : background - color 250ms ؛ opacity : 1 ؛ transition : opacity 250ms ؛ webkit - transition : opacity 250ms ؛ } .ctaText { font - weight : bold ؛ color : # 3498DB ؛ text - decoration : none ؛ font - size : 16px ؛ } .postTitle { color : # 34495E ؛ text - decoration : underline!important ؛ font - size : 16px ؛ } : hover .postTitle { text - decoration : underline!important ؛ } ', re.DOTALL)
            pattern = re.compile(r'\.related - post h3 \{.*?\}\s*\.yarpp - related \.minientry \{.*?\}\s*\.yarpp - related \.post - info \{.*?\}\s*\.yarpp - related a \{.*?\}\s*', re.DOTALL)
            text = re.sub(pattern, ' ' , text)
            text = self.clean_cultura(text)
            
        text = self.symbol_pattern.sub(' ', text)
        text = self.source_web_pattern.sub('', text)
        text = re.sub(r'(?<=[^\s\n])#(?=[^\s\n])', '', text)
        text = re.sub(r'@\w+', '', text)
        
        return text

    '''remove lines with character numbers more than a thereshold'''
    def remove_numeric_lines(self, line):
        num_num_count = sum(1 for char in line if char.isdigit() or char in ':><؟!.،,?..,?!;:-()[]{}"\'')
        total_chars = len(line)
        if num_num_count / total_chars >= self.numeric_lines_thresh:
            return ""
        else:
            return line

    '''remove lines with symbol characters more than a thereshold
        Arabic characters and symbols in the Unicode ranges:
        \u0600-\u06FF: Arabic
        \u0750-\u077F: Arabic Supplement
        \u08A0-\u08FF: Arabic Extended-A
        \uFB50-\uFDFF: Arabic Presentation Forms-A
        \uFE70-\uFEFF: Arabic Presentation Forms-B
        General punctuation and symbols in the Unicode ranges:
        \u2000-\u206F: General Punctuation
        \u2E00-\u2E7F: Supplemental Punctuation
        ASCII punctuation and symbols: !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`'''
    def remove_symbolic_lines(self, line):
        pattern = r'[\uF000-\uF0FF]|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u2000-\u206F\u2E00-\u2E7F!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
        matches = re.findall(symbol_regex, line) #pattern
        num_punct_count = len(matches)
        total_chars = len(line)
        if num_punct_count / total_chars >= self.symbolic_lines_thresh:
            return ""
        else:
            return line

    
    def remove_english_symbolic_lines(self, line):
        # css_pattern = re.compile(
        #     r'[.#]?\w+\s*-\s*\w+\s*{\s*[\w\s:-]+;\s*|\w+\s*:\s*[^;]+؛|;|\(.*?\)\s*؛|#\w+\s*{\s*[^}]+؛|;|\d+px'
        # )
        # if not bool(css_pattern.search(line)):
        #     print(line)
        #     return ""
        # Define the regex pattern to match any CSS block
        # if re.search(r'<\s*\/?\s*[a-zA-Z][a-zA-Z0-9]*\s*[^>]*>', line): #r'<[^>]+>'
        # # if re.search(pattern, line):
        #     reshaped_text = arabic_reshaper.reshape(line)
        #     bidi_text = get_display(reshaped_text)
        #     print(bidi_text)
        #     # print('html')
        #     return ""
        # # Check if the line looks like a CSS rule (selectors and properties)
        # if re.search(r'^\s*[a-zA-Z-]+\s*:\s*[^;]+;\s*$', line):
        #     reshaped_text = arabic_reshaper.reshape(line)
        #     bidi_text = get_display(reshaped_text)
        #     print(bidi_text)
        #     # print('css')
        #     return ""
        
        if re.search(r'{[^}]+}[^}]*}[^}]*}[^}]*}', line):
            # reshaped_text = arabic_reshaper.reshape(line)
            # bidi_text = get_display(reshaped_text)
            # print(bidi_text)
            return ""
        return line
    
    
    '''remove ip, username, phone number and emails'''
    def remove_personal_info(self, text):
        combined_patterns = r"|".join([self.id_patterns, self.key_pattern, self.ip_pattern])
        text = re.sub(combined_patterns, '', text)
        
        '''Replace matched email addresses with 'email@domain.com\''''
        text = self.email_pattern.sub('email@domain.com', text)
        text = re.sub(r'[\w\.\+]+email@domain.com', 'email@domain.com', text)
        text = re.sub(self.shaba_pattern, 'IR000000000000000000000000', text)
        text = re.sub(self.account_number_pattern, '0000-0000-0000-0000', text)

        return text

    '''remove emojis present in text'''
    def remov_emojis(self, text):
        # text = self.emoji_chars_codes.sub(r'', text) #corrupts some files
        return emoji.replace_emoji(text, replace=' ')
    
    def normalize_dates(self, text):
        text = text.replace('ه . ش', 'ه.ش').replace('ه . ق', 'ه.ق')
        return text

    '''normalize non-English/Araboc/Persian numbers with Persian numbers'''
    def normalize_numbers(self, text):
        '''Define the mappings for Persian numbers'''
        persian_numbers = {
            '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
            '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
        }
    
        '''Helper function to detect if a character is a number'''
        def is_number(char):
            return unicodedata.category(char).startswith('N')
    
        '''Helper function to convert a number to Persian if it's not English, Persian, or Arabic'''
        def convert_to_persian(char):
            if '0' <= char <= '9':  # English numbers
                return char
            elif '۰' <= char <= '۹':  # Persian numbers
                return char
            elif '٠' <= char <= '٩':  # Arabic numbers
                return char
            else:
                # Convert other numbers to Persian
                try:
                    num = int(unicodedata.digit(char))
                    return persian_numbers[str(num)]
                except (ValueError, KeyError):
                    return char
        
        '''Replace non-English, non-Persian, and non-Arabic numbers'''
        text = ''.join(convert_to_persian(char) if is_number(char) else char for char in text)
        return text

    '''normalize non-English/Araboc/Persian symbols/punctuations with Persian symbols/punctuations'''
    def normalize_symbols(self, text):
        '''Define the mappings for Persian punctuations'''
        persian_punctuation = {
            '!': '!', ',': '،', ';': '؛', '?': '؟',
            '.': '.', ':': ':', '"': '«', "'": '»',
            '(': '(', ')': ')', '-': '-', '_': '_'
        }
        '''Helper function to detect if a character is punctuation'''
        def is_punctuation(char):
            return unicodedata.category(char).startswith('P')
        
        '''Helper function to convert punctuation to Persian'''
        def convert_punctuation_to_persian(char):
            if char in persian_punctuation:
                return persian_punctuation[char]
            return char
        
        '''Replace non-English, non-Persian, and non-Arabic numbers and punctuations'''
        text = ''.join(convert_punctuation_to_persian(char) if is_punctuation(char) else char for char in text)
        
        return text