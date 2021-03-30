import os
import re
import unidecode
import unicodedata
import numpy as np
from cytoolz import curry
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from src.com.pre_train.base_data import TxtLmdb
PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../"))
LOG_DIR = PROJ_DIR + '/log'
IMAGE_HEIGHT = 72
IMAGE_WIDTH = 96
IMAGE_DEPTH = 64

@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids
@curry
def open_lmdb(db_dir, readonly=False):
    # print('entering:')
    db = TxtLmdb(db_dir, readonly)
    # try:
    return db

def preprocess(text):
    # lower case
    control_char_regex = re.compile(r'[\r\n\t]+')

    text = text.lower()
    text = text.replace('\n','')
    # replace USER TAG
    # text = re.sub(r'(^|[^@\w])@(\w{1,15})\b', '@USER', text)
    # ^@[A-Za-z0-9_]{1,15}$
    # replace HTTP tag
    text = re.sub(r"http\S+", "URL", text)

    # text = html.unescape(text)
    # text = text.translate(transl_table)
    text = text.replace('…', '...')
    text = re.sub(control_char_regex, ' ', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = ' '.join(text.split())
    text = text.strip()
    # demojize
    # text = emoji.demojize(text)
    text = unidecode.unidecode(text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')

    return text
class DictObj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [DictObj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, DictObj(b) if isinstance(b, dict) else b)


def get_word(text, i):
    """
            returns the first word in a given text.
        """
    text = text.replace('.', ' ').replace(',', ' ').strip()
    return text.split()[i]


# 计算最大共同字串：
def getMaxCommonSubstr(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    p = 0  # 匹配的起始位

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    return count_words(str1[p - maxNum:p]) + 1, str1[p - maxNum:p]

def printMatrixList(li):
    # 打印多维list
    row = len(li)
    col = len(li[0])

    for i in range(row):
        for j in range(col):
            print(li[i][j], end=' ')
        print('')


def count_words(s):
    count = len(s.split())
    return count


s = StandardScaler()


def standardization(data, type):
    if type == 'valid':
        # 验证集 直接用transform
        return s.transform(
            data.astype(np.float32).reshape(-1, 1)).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])
    else:
        return s.fit_transform(
            data.astype(np.float32).reshape(-1, 1)).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])


def noramlization(data):
    import numpy as np
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    # normData = np.zeros(np.shape(data))
    # m = normData.shape[0]
    normData = data - np.tile(minVals, np.shape(data))
    normData = normData / np.tile(ranges, np.shape(data))
    return normData, ranges, minVals


def _parse_function(example_proto):
    feature_description = {
        'vol_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
        # 'sub-id': tf.io.FixedLenFeature([], tf.string)
    }
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


def generate_array_from_file(filenames, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH):
    volume_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
    parallel_work = 1
    X = []
    Y = []
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    parsed_dataset = dataset.map(_parse_function, num_parallel_calls=parallel_work)
    iterator = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)
    cnt = 0

    try:
        while True:
            cnt = cnt + 1
            parsed_record = iterator.get_next()

            # type:  'tensorflow.python.framework.ops.EagerTensor'
            feature = tf.compat.v1.decode_raw(parsed_record['vol_raw'], tf.float32)
            label = parsed_record['label']
            volume = tf.reshape(feature, volume_shape)
            a = volume.numpy()
            # if cnt <= 10:
            #     continue
            X.append(a)
            Y.append(label)
    except tf.errors.OutOfRangeError:
        return X, Y
        pass


def chmod(path, mode):
    import re
    import os
    import stat

    RD, WD, XD = 4, 2, 1
    BNS = [RD, WD, XD]
    MDS = [
        [stat.S_IRUSR, stat.S_IRGRP, stat.S_IROTH],
        [stat.S_IWUSR, stat.S_IWGRP, stat.S_IWOTH],
        [stat.S_IXUSR, stat.S_IXGRP, stat.S_IXOTH]
    ]
    if isinstance(mode, int):
        mode = str(mode)
    if not re.match("^[0-7]{1,3}$", mode):
        raise Exception("mode does not conform to ^[0-7]{1,3}$ pattern")
    mode = "{0:0>3}".format(mode)
    mode_num = 0
    for midx, m in enumerate(mode):
        for bnidx, bn in enumerate(BNS):
            if (int(m) & bn) > 0:
                mode_num += MDS[bnidx][midx]
    os.chmod(path, mode_num)


def make_print_to_file(path=LOG_DIR):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import os
    # import config_file as cfg_file
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day and time:' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # print -> log
    #############################################################
    print(fileName.center(60, '*'))
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
# def sentence_split(str_centence):
#     list_ret = list()
#     for s_str in str_centence.split('.'):
#         if '?' in s_str:
#             list_ret.extend(s_str.split('?'))
#         elif '!' in s_str:
#             list_ret.extend(s_str.split('!'))
#         else:
#             list_ret.append(s_str+'.')
#     return list_ret

if __name__ == '__main__':
    _list = split_into_sentences(preprocess('Henry Hallam (July 9, 1777 - January 21, 1859) was an  English historian.'
                         'The only son of John Hallam, canon of Windsor and dean of Bristol, '
                         'he was educated at Eton and Christ Church, Oxford, graduating in 1799. '
                         'Called to the bar, he practised for some years on the Oxford circuit; '
                         'but his tastes were literary, and when, on his father\'s death in 1812, he '
                         'inherited a small estate in Lincolnshire, he gave himself up wholly to academic'
                         ' study. He had become connected with the brilliant band of authors and politicians '
                         'who led the Whig party, a connection to which he owed his appointment to the well-paid'
                         ' and easy post of commissioner of stamps; but took no part in politics himself.  He was,'
                         ' however, an active supporter of many popular movements--particularly of that which ended'
                         ' in the abolition of the slave trade; and he was sincerely attached to the political principles'
                         ' of the Whigs. Hallam\'s earliest literary work was undertaken in connexion with the great organ '
                         'of the Whig party, the Edinburgh Review, where his review of Scott\'s Dryden attracted attention. '
                         'His first great work, The View of the State of Europe during the Middle Ages, '
                         'was produced in 1818, and was followed nine years later by the Constitutional '
                         'History of England. In 1838-1839 appeared the Introduction to the Literature of '
                         'Europe in the 15th, 16th and 17th Centuries.  These are the three works on which '
                         'Hallam\'s fame rests. They took a place in English literature which was not seriously'
                         ' challenged until the 20th century. A volume of supplemental notes to his Middle Ages '
                         'was published in 1848; these facts and dates represent nearly all of Hallam\'s career. '
                         'The strongest personal interest in his life was the affliction which befell him in the loss of his children, one after another. His eldest son, '
                                      'Arthur Henry Hallam--the "A.H.H." of Tennyson'
                         '\'s In Memoriam, and by the testimony of his contemporaries a man of the most brilliant promise--'
                         'died in 1833 at the age of twenty-two. Seventeen years later, his second son, Henry Fitzmaurice'
                         ' Hallam, was cut off like his brother at the very threshold of what might have been a great career'
                         '. The premature death and high talents of these young men, and the association of one of them with'
                         ' the most popular poem of the age, have made Hallam\'s family afflictions better known '
                         'than any other incidents of his life. He survived wife, daughter and sons by many years'
                         '.In 1834 Hallam published The Remains in Prose and Verse of Arthur Henry Hallam, with '
                         'a Sketch of his Life. In 1852 a selection of Literary Essays and Characters from '
                         'the Literature of Europe was published. Hallam was a fellow of the Royal Society, '
                         'and a trustee of the British Museum, and enjoyed many other appropriate distinctions. '
                         'In 1830 he received the gold medal for history, founded by George IV.  The Middle Ages '
                         'is described by Hallam himself as a series of historical dissertations, a comprehensive '
                         'survey of the chief circumstances that can interest a philosophical inquirer during the '
                         'period from the 5th to the 15th century. The work consists of nine long chapters, each of '
                         'which is a complete treatise in itself. The history of France, of Italy, of Spain, of'
                         ' Germany, and of the Greek and Saracenic empires, sketched in rapid and general terms, '
                         'is the subject of five separate chapters. Others deal with the great institutional '
                         'features of medieval society--the development of the feudal system, of the '
                         'ecclesiastical system, and of the free political system of England. The last'
                         ' chapter sketches the general state of society, the growth of commerce, manners,'
                         ' and literature in the Middle Ages. The book may be regarded as a general view of'
                         ' early modern history, preparatory to the more detailed treatment of special lines of'
                         ' inquiry carried out in his subsequent works, although Hallam\'s original intention '
                         'was to continue the work on the scale on which it had been begun.The Constitutional '
                         'History of England takes up the subject at the point at which it had been dropped '
                         'in the View of the Middle Ages, viz, the accession of Henry VII, and carries'
                         ' it down to the accession of George III. Hallam stopped here for a '
                         'characteristic reason, which it is impossible not to respect and '
                         'to regret. He was unwilling to excite the prejudices of modern politics which'
                         ' seemed to him to run back through the whole period of the reign of George III; nevertheless, '
                         'he was accused of bias.  The Quarterly Review for 1828 contains an article on the '
                         'Constitutional History, written by Southey, full of reproach. The work, he says. '
                         'is the "production of a decided partisan," who "rakes in the ashes of long-forgotten '
                         'and a thousand times buried slanders, for the means of heaping obloquy on all who '
                         'supported the established institutions of the country." Hallam\'s view of '
                         'constitutional history was that it should contain only so much of the '
                         'political and general history of the time as bears directly on specific '
                         'changes in the organization of the state, including judicial as well as '
                         'ecclesiastical institutions.  It was his cool treatment of such sanctified'
                         ' names as Charles I, Cranmer and Laud that provoked the indignation of '
                         'Southey, who forgot that the same impartial measure was extended '
                         'to statesmen on the other side.If Hallam ever deviated'
                         ' from perfect fairness, it was in the tacit assumption '
                         'that the 19th century theory of the constitution was the '
                         'right theory in previous centuries, and that those who departed'
                         ' from it on one side or the other were in the wrong. He'
                         ' did unconsciously antedate the constitution, and it is clear'
                         ' from incidental allusions in his last work that he did'
                         ' not favour the democratic changes he thought to be impending.'
                         ' Hallam, like Macaulay, ultimately referred all political questions to the'
                         ' standard of Whig constitutionalism. But he was scrupulously conscientious in collecting and '
                         'weighing his materials. In this he was helped by his legal training, and it was this which made the Constitutional '
                         'History one of the standard text-books of English politics.Like the Constitutional History, the Introduction to'
                         ' the Literature of Europe continues a branch of inquiry which had been opened in the View of the Middle Ages. '
                         'In the first chapter of the Literature, which is to a great extent supplementary to the last chapter of the Middle Ages, '
                         'Hallam sketches the state of literature in Europe down to the end of the 14th century: the extinction of ancient learning '
                         'which followed the fall of the Roman empire and the rise of Christianity; the preservation of the Latin language in the services '
                         'of the church; and the slow revival of letters, which began to show itself soon after the 7th century--"the nadir of the human mind"--'
                         'had been passed. For the first century and a half of his special period he is mainly occupied with a review of classical learning, and '
                         'he adopts the plan of taking short decennial periods and noticing the most remarkable works which they produced.  The rapid growth of'
                         ' literature in the 16th century compels him to resort to a classification of subjects: in the period 1520-1550 we have separate chapters'
                         ' on ancient literature, theology, speculative philosophy and jurisprudence, the literature of taste, and scientific and miscellaneous '
                         'literature; and the subdivisions of subjects is carried '
                         'further of course in the later periods. Thus poetry, the drama and polite literature form the subjects'
                         ' of separate chapters. One inconvenient result of this arrangement is that the same author is scattered '
                         'over many chapters, according as his works fall within this category or that period of time. Names like '
                         'Shakespeare, Grotius, Francis Bacon and Thomas Hobbes appear in half a dozen different places. The '
                         'individuality of great authors is thus dissipated except when it has been preserved by an occasional '
                         'sacrifice of the arrangement--and this defect, if it is to be esteemed a defect, is increased by the '
                         'very sparing references to personal history and character with which Hallam was obliged to content himself '
                         'His plan excluded biographical history, nor is the work, he tells us, to be regarded as one of reference. '
                         'It is rigidly anaccount of the books which would make a complete library of the period, arranged according to the'
                         ' date of their publication and the nature of their subjects. The history '
                         'of institutions like universities and academies, and that of great popular movements like the Reformation,'
                         ' are of course noticed in their immediate connection with literary results; but Hallam had little taste for'
                         ' the spacious generalization which such subjects suggest. The great qualities displayed in this work have been'
                         ' universally acknowledged--conscientiousness, accuracy, judgment and enormous reading. Not the least styiking testimony to Hallam\'s '
                         'powers is his mastery over so many diverse forms of intellectual activity. In science and theology, mathematics '
                         'and poetry, metaphysics and law, he is a competent and always a fair if not a profound critic. The bent of his '
                         'own mind is manifest in his treatment of pure literature and of political speculation--which seems to be inspired with'
                         ' stronger personal interest and a higher sense of power than other parts of his work display. Not less worthy of notice '
                         'in a literary history is the good sense by which both his learning and his tastes have been held in control. Probably no writer'
                         ' ever possessed a juster view of the relative importance of men and things. The labour devoted to an investigation is with Hallam no'
                         ' excuse for dwelling on the result, unless that is in itself important. He turns away contemptuously from the mere curiosities of literature,'
                         ' and is never tempted to make a display of trivial erudition. Nor do we find that his interest in special studies leads him to assign them a disproportionate place in his general view of the literature of a period.Hallam is generally described as a "philosophical historian." The description is justified not so much by any philosophical quality in his method as by the nature of his subject and his own temper. Hallam is a philosopher to this extent that both in political and in literary history he fixed his attention on results rather than on persons. His conception of history embraced the whole movement of society. Beside that conception the issue of battles and the fate of kings fall into comparative insignificance. "We can trace the pedigree of princes," he reflects, "fill up the catalogue of towns besieged and provinces desolated, describe even the whole pageantry of coronations and festivals, but we cannot recover the genuine history of mankind." '
                         'But, on the other hand, there is no trace in Hallam of anything like a philosophy of history or '
                         'society.Wise and generally melancholy reflections on human nature and political society are not '
                         'infrequent in his writings, and they arise naturally and incidentally out of the subject he is discussing. '
                         'His object is the attainment of truth in matters of fact. Sweeping theories of the movement of society, '
                         'and broad characterizations of particular periods of history seem to have no attraction for him.  '
                         'The view of mankind on which such generalizations are usually based, taking little account of individual '
                         'character, was distasteful to him. Thus he objects to the use of statistics because they favour the tendency'
                         ' to regard all men as mentally and morally equal. At the same time Hallam by no means assumes the tone of the'
                         ' mere scholar. He is solicitous to show that his point of view is that of the cultivated gentleman and not of '
                         'the specialist. Thus he tells us that Montaigne is the first French author whom an English gentleman is ashamed '
                         'not to have read. In fact, allusions to the necessary studies of a gentleman meet us constantly, reminding us of '
                         'the unlikely erudition of the schoolboy in Macaulay. Hallam\'s prejudices, so far as he had '
                         'any, belong to the same character. His criticism assumes a tone of moral censure when he has '
                         'to deal with certain extremes of human thought--scepticism in philosophy, atheism in religion'
                         ' and democracy in politics.Macaulay\'s essay in review of the Constitutional History is available '
                         'at: http://www.history1700s.com/etexts/html/texts/1cahe10.txt References.;'))
    list_i = []
    list_i.extend(_list)
    # for i in _list:
    #     print(i)
    #     print("***********************************")
    print(len(list_i))

