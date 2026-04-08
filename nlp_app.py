import streamlit as st
import re
import spacy
from spacy import displacy
import torch
import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')

DISCOURSE_MARKERS = {
    'Temporal': ['when', 'after', 'before', 'while', 'since', 'until', 'as', 'once', 'whenever'],
    'Contingency': ['because', 'since', 'so', 'therefore', 'thus', 'hence', 'consequently', 'as a result'],
    'Comparison': ['but', 'however', 'although', 'though', 'whereas', 'while', 'yet', 'nevertheless'],
    'Expansion': ['and', 'or', 'also', 'moreover', 'furthermore', 'besides', 'in addition', 'additionally']
}

def rule_based_edu_segmentation(text):
    doc = nlp(text)
    edus = []
    current_edu = []
    boundary_tokens = []
    
    for i, token in enumerate(doc):
        current_edu.append(token.text)
        
        is_boundary = False
        
        if token.text in ['.', '!', '?', ';']:
            is_boundary = True
            boundary_tokens.append(token.text)
        
        if token.pos_ == 'SCONJ' and token.text.lower() in ['although', 'because', 'since', 'while', 'when', 'if', 'unless']:
            if current_edu and len(current_edu) > 1:
                is_boundary = True
                boundary_tokens.append(token.text)
        
        if token.dep_ == 'mark' and token.head.pos_ == 'VERB':
            if len(current_edu) > 2:
                is_boundary = True
                boundary_tokens.append(token.text)
        
        if is_boundary and len(current_edu) > 0:
            edu_text = ' '.join(current_edu)
            edus.append({'text': edu_text, 'boundary': token.text})
            current_edu = []
    
    if current_edu:
        edu_text = ' '.join(current_edu)
        edus.append({'text': edu_text, 'boundary': None})
    
    return edus, boundary_tokens

def find_discourse_markers(sentence):
    sentence_lower = sentence.lower()
    found_markers = []
    
    for category, markers in DISCOURSE_MARKERS.items():
        for marker in markers:
            pattern = r'\b' + re.escape(marker) + r'\b'
            matches = list(re.finditer(pattern, sentence_lower))
            for match in matches:
                found_markers.append({
                    'marker': match.group(),
                    'category': category,
                    'start': match.start(),
                    'end': match.end()
                })
    
    return sorted(found_markers, key=lambda x: x['start'])

def extract_arguments(sentence, marker_info):
    marker = marker_info['marker']
    start = marker_info['start']
    
    arg1 = sentence[:start].strip()
    arg2 = sentence[start + len(marker):].strip()
    
    if arg1.endswith(','):
        arg1 = arg1[:-1].strip()
    if arg2.startswith(','):
        arg2 = arg2[1:].strip()
    
    return arg1, arg2

@st.cache_resource
def load_coref_model():
    try:
        import spacy
        nlp_coref = spacy.load('en_core_web_sm')
        return nlp_coref
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        return None

def perform_coreference(text, nlp_model):
    try:
        doc = nlp_model(text)
        
        pronouns = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
        
        male_pronouns = ['he', 'him', 'his']
        female_pronouns = ['she', 'her', 'hers']
        neutral_pronouns = ['it', 'its']
        
        male_names = ['barack', 'obama', 'donald', 'trump', 'joe', 'biden', 'bill', 'clinton', 'george', 'bush', 
                      'john', 'michael', 'david', 'james', 'robert', 'william', 'richard', 'thomas', 'charles']
        female_names = ['hillary', 'clinton', 'michelle', 'obama', 'angela', 'merkel', 'theresa', 'may', 
                        'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'susan', 'jessica', 'sarah']
        
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_,
                    'gender': guess_gender(ent.text, male_names, female_names)
                })
        
        clusters = []
        
        for token in doc:
            if token.text.lower() in pronouns:
                closest_entity = None
                min_distance = float('inf')
                
                for entity in entities:
                    if token.idx > entity['end']:
                        distance = token.idx - entity['end']
                    else:
                        continue
                    
                    if token.text.lower() in male_pronouns:
                        if entity['label'] == 'PERSON' and entity['gender'] in ['male', 'unknown']:
                            if distance < min_distance:
                                min_distance = distance
                                closest_entity = entity
                    elif token.text.lower() in female_pronouns:
                        if entity['label'] == 'PERSON' and entity['gender'] in ['female', 'unknown']:
                            if distance < min_distance:
                                min_distance = distance
                                closest_entity = entity
                    elif token.text.lower() in neutral_pronouns:
                        if entity['label'] in ['ORG', 'GPE']:
                            if distance < min_distance:
                                min_distance = distance
                                closest_entity = entity
                
                if closest_entity:
                    clusters.append([closest_entity, {'text': token.text, 'start': token.idx, 'end': token.idx + len(token.text)}])
        
        merged_clusters = {}
        for cluster in clusters:
            entity_text = cluster[0]['text']
            if entity_text not in merged_clusters:
                merged_clusters[entity_text] = [cluster[0]]
            
            if cluster[1] not in merged_clusters[entity_text]:
                merged_clusters[entity_text].append(cluster[1])
        
        return list(merged_clusters.values())
        
    except Exception as e:
        import traceback
        st.error(f"指代消解错误: {str(e)}")
        st.code(traceback.format_exc())
        return []

def guess_gender(name, male_names, female_names):
    name_lower = name.lower()
    
    if any(male_name in name_lower for male_name in male_names):
        return 'male'
    elif any(female_name in name_lower for female_name in female_names):
        return 'female'
    
    return 'unknown'

def main():
    st.set_page_config(page_title='NLP高级任务演示', page_icon='🔬', layout='wide')
    
    st.title('🔬 自然语言处理高级任务演示系统')
    st.markdown('---')
    
    tab1, tab2, tab3 = st.tabs(['📝 EDU切分', '🔗 浅层篇章分析', '👥 指代消解'])
    
    with tab1:
        st.header('话语分割 (EDU切分)')
        st.markdown('**基于规则的EDU切分演示**')
        
        default_text = "The company announced record profits. This was due to strong sales in Europe. However, Asian markets showed weaker performance."
        
        input_sentence = st.text_area('输入句子（可以是多个句子）', default_text, height=100, key='edu_input')
        
        if st.button('🔍 进行EDU切分', key='analyze_edu'):
            if input_sentence.strip():
                clean_text = re.sub(r'\s+', ' ', input_sentence)
                rule_edus, boundaries = rule_based_edu_segmentation(clean_text)
                
                st.subheader('📊 EDU切分结果')
                st.markdown(f'**检测到 {len(boundaries)} 个边界标记，共切分为 {len(rule_edus)} 个EDU**')
                
                for i, edu in enumerate(rule_edus, 1):
                    if edu['boundary']:
                        st.markdown(
                            f'<div style="border: 2px solid #4CAF50; padding: 10px; margin: 5px 0; border-radius: 5px; background-color: #f1f8e9;">'
                            f'<strong>EDU {i}:</strong> {edu["text"]} '
                            f'<span style="background-color: #FFD700; padding: 2px 6px; border-radius: 3px; font-weight: bold;">'
                            f'🎯 边界: {edu["boundary"]}</span></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div style="border: 2px solid #2196F3; padding: 10px; margin: 5px 0; border-radius: 5px; background-color: #e3f2fd;">'
                            f'<strong>EDU {i}:</strong> {edu["text"]}</div>',
                            unsafe_allow_html=True
                        )
                
                with st.expander('ℹ️ 切分规则说明'):
                    st.markdown("""
                    **启发式规则：**
                    1. 句末标点 (. ! ? ;) 后切分
                    2. 从属连词 (although, because, since等) 前切分
                    3. 依存关系为 'mark' 的动词前切分
                    
                    **边界标记说明：**
                    - 🎯 黄色标签表示检测到的边界词
                    - 绿色边框表示以边界词结束的EDU
                    - 蓝色边框表示最后一个EDU（无明确边界）
                    """)
            else:
                st.warning('请输入句子')
    
    with tab2:
        st.header('浅层篇章分析')
        st.markdown('**基于显式连接词的论据提取**')
        
        default_sentence = "Third-quarter sales in Europe were exceptionally strong, boosted by promotional programs and new products - although weaker foreign currencies reduced the company's earnings."
        
        sentence = st.text_area('输入句子', default_sentence, height=100, key='discourse_input')
        
        if st.button('🔍 分析篇章结构', key='analyze_discourse'):
            markers = find_discourse_markers(sentence)
            
            if markers:
                st.subheader('📊 连接词识别结果')
                
                highlighted_sentence = sentence
                offset = 0
                
                for marker in reversed(markers):
                    original_text = sentence[marker['start']:marker['end']]
                    highlighted = f'<strong style="color: white; background-color: #FF5722; padding: 2px 8px; border-radius: 4px;">{original_text}</strong> <span style="background-color: #FFCDD2; padding: 2px 6px; border-radius: 3px; font-size: 12px;">[{marker["category"].upper()}]</span>'
                    
                    start = marker['start'] + offset
                    end = marker['end'] + offset
                    highlighted_sentence = highlighted_sentence[:start] + highlighted + highlighted_sentence[end:]
                    offset += len(highlighted) - (marker['end'] - marker['start'])
                
                st.markdown(f'<div style="background-color: #F5F5F5; padding: 15px; border-radius: 5px; line-height: 2;">{highlighted_sentence}</div>', unsafe_allow_html=True)
                
                st.subheader('📑 论据提取')
                
                for i, marker in enumerate(markers, 1):
                    st.markdown(f'**连接词 {i}:** `{marker["marker"]}` ({marker["category"]})')
                    
                    arg1, arg2 = extract_arguments(sentence, marker)
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown(
                            f'<div style="background-color: #E3F2FD; padding: 15px; border-radius: 5px; border-left: 4px solid #2196F3;">'
                            f'<strong>Arg1 (前置论据):</strong><br>{arg1}</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col_b:
                        st.markdown(
                            f'<div style="background-color: #FFF3E0; padding: 15px; border-radius: 5px; border-left: 4px solid #FF9800;">'
                            f'<strong>Arg2 (后置论据):</strong><br>{arg2}</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.warning('未检测到显式连接词')
            
            with st.expander('ℹ️ 支持的连接词类别'):
                for category, markers_list in DISCOURSE_MARKERS.items():
                    st.markdown(f'**{category}:** {", ".join(markers_list)}')
    
    with tab3:
        st.header('指代消解')
        st.markdown('**基于spaCy的启发式指代消解**')
        
        default_text = """Barack Obama was born in Hawaii. He served as the 44th president of the United States. Obama was the first African-American president. His presidency lasted from 2009 to 2017. During his term, he signed many important laws."""
        
        input_text = st.text_area('输入文本', default_text, height=150, key='coref_input')
        
        if st.button('🔍 执行指代消解', key='analyze_coref'):
            with st.spinner('正在分析...'):
                try:
                    model = load_coref_model()
                    
                    if model is None:
                        st.error('模型加载失败')
                    else:
                        clusters = perform_coreference(input_text, model)
                        
                        if clusters:
                            st.subheader('🎯 指代簇识别结果')
                            
                            colors = ['#FFEbee', '#E3F2FD', '#E8F5E9', '#FFF3E0', '#F3E5F5', '#E0F7FA', '#FBE9E7', '#F1F8E9']
                            
                            cluster_info = []
                            
                            for i, cluster in enumerate(clusters):
                                mentions = [m['text'] for m in cluster]
                                cluster_info.append({
                                    'cluster_id': i + 1,
                                    'mentions': mentions,
                                    'color': colors[i % len(colors)]
                                })
                            
                            highlighted_text = input_text
                            for info in reversed(cluster_info):
                                for mention in info['mentions']:
                                    if mention in highlighted_text:
                                        highlighted_text = highlighted_text.replace(
                                            mention,
                                            f'<span style="background-color: {info["color"]}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{mention}</span>',
                                            1
                                        )
                            
                            st.markdown('**高亮文本:**')
                            st.markdown(
                                f'<div style="background-color: #FAFAFA; padding: 20px; border-radius: 5px; line-height: 2.5; font-size: 16px;">{highlighted_text}</div>',
                                unsafe_allow_html=True
                            )
                            
                            st.subheader('📋 指代链列表')
                            
                            for info in cluster_info:
                                mentions_str = ', '.join([f'"{m}"' for m in info['mentions']])
                                st.markdown(
                                    f'<div style="background-color: {info["color"]}; padding: 10px; margin: 5px 0; border-radius: 5px;">'
                                    f'<strong>Cluster {info["cluster_id"]}:</strong> [{mentions_str}]</div>',
                                    unsafe_allow_html=True
                                )
                            
                            with st.expander('ℹ️ 说明'):
                                st.markdown('''
                                **启发式规则：**
                                - 识别命名实体 (PERSON, ORG, GPE)
                                - 根据性别匹配代词 (he/him/his → 男性, she/her → 女性)
                                - 根据类型匹配代词 (it → ORG/GPE)
                                ''')
                        else:
                            st.warning('未找到指代关系')
                
                except Exception as e:
                    st.error(f'发生错误: {str(e)}')

if __name__ == '__main__':
    main()
