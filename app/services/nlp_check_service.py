from collections import Counter
from konlpy.tag import Kkma, Okt
from kiwipiepy import Kiwi
import re

# 불필요한 추임새 분석
def filler_words_check(txt):
    filler_words = ['설마', '그렇군요', '그렇구나', '그럼', '아야', '마구', '그러니까', '말하자면', '그다지', '어머나', '맞아요', '저', '있잖아', '아', '그래', '뭐랄까', '그', '뭐라고', '글쎄', '솔직히', '뭐지', '뭐더라', '그래요', '아무튼', '에이', '막', '아이고', '예', '어머', '세상에', '자', '뭐', '우와', '그게', '글쎄요', '정말', '음', '맞아', '어쨌든', '좀', '야', '진짜', '별로', '네', '참', '에휴', '쉿', '어', '저기요', '그냥']
    # KoNLPy의 Okt를 사용하여 한국어 텍스트를 토큰화 (단어 단위로 분리)
    okt = Okt()
    word_tokens = okt.morphs(txt)  # word_tokenize 대신 okt.morphs 사용
    result = [word for word in word_tokens if word in filler_words]
    count = Counter(result).most_common()
    return [{'text': c[0], 'weight': c[1]} for c in count]

# 어미 분석
def word_end_check(txt):

    kkma = Kkma()
    pos = kkma.pos(txt)
    count = Counter(pos)
    word_a = sum(count[i] for i in count if i[1] in ('EFQ', 'EFA'))  # 의문, 청유형
    word_b = sum(count[i] for i in count if i[1] in ('EFN', 'EFR'))  # 평서, 존칭형
    rate1 = word_a / (word_a + word_b) * 100  # 참여유도형 비율
    rate2 = word_b / (word_a + word_b) * 100  # 공식적인 화법 비율
    return {'formal_speak': int(rate2), 'question_speak': int(rate1)}

# 명사 리스트 추출 (워드 클라우드 생성용)
def get_nouns_list(txt):
    # print('원래 문장:',txt)
    # okt = Okt()
    # nouns = okt.nouns(txt)
    # nouns_count = Counter(nouns).most_common()
    # return [{'text': n[0], 'weight': n[1]} for n in nouns_count]


    # word_extractor = WordExtractor()
    # words = word_extractor.train_extract([txt])  # 단어 추출
    # scores = {word: score.cohesion_forward for word, score in words.items()}
    # tokenizer = LTokenizer(scores=scores)
    #
    # tokens = tokenizer.tokenize(txt)
    # words_count = Counter(tokens).most_common()
    # return [{'text': w[0], 'weight': w[1]} for w in words_count]
    kiwi = Kiwi()
    result = kiwi.analyze(txt)
    words = [token.form for token in result[0][0] if token.tag.startswith('N')]  # 명사만 추출
    words_count = Counter(words).most_common()
    return [{'text': w[0], 'weight': w[1]} for w in words_count]

# 메인 실행 함수 (텍스트 분석)
def text_analysis(text):
    filler_words = filler_words_check(text) #불필요한 추임새
    print('filler_words_check')

    speak_end = word_end_check(text) #어미 분석(?!.)
    print('word_end_check')

    word_list = get_nouns_list(text) #워드 클라우드용(단어집합)
    print('get_nouns_list')

    fillertext_values = ', '.join([w['text'] for w in filler_words])
    fillerweight_values = ', '.join([str(w['weight']) for w in filler_words])

    text_values = ', '.join([w['text'] for w in word_list])
    weight_values = ', '.join([str(w['weight']) for w in word_list])


    text_json = {
        "fillerwords": fillertext_values,
        "fillerweights":fillerweight_values,
        "speak_end": speak_end,
        "text":text_values,
        "weight": weight_values
    }
    return text_json
