import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import yaml
import os
import numpy as np
import subprocess
import pandas as pd

def remove_duplicate(results_list, k_intersecion = 0.5):
    '''
    Боремся с дублированием. В случае обнаружения нескольких одинаковых классов, 
    проверяем, нет ли пересечений их площадей. Если пересечение площадей составляет более 50%
    от площади наименьшей области, присваиваем id_class = -1
    results_list - результат детектирования, список из x1, y1, x2, y2, score, class_id
    k_intrsection - коэффициент пересечения прямоугольников, доля от площади минимального по площади прямоугольника
    из двух дубликатов
    '''
    for i in range(0, len(results_list)-1):
        x1, y1, x2, y2, score, class_id = results_list[i]
        square = abs(x1-x2) * abs(y1-y2)
        for j in range(i+1, len(results_list)):
            x1_, y1_, x2_, y2_, score_, class_id_ = results_list[j]
            if (class_id == class_id_):
                square_ = abs(x1_-x2_) * abs(y1_-y2_)
                Left = max(x1, x1_)
                Top = max(y1, y1_)
                Right = min(x2, x2_)
                Bottom = min(y2, y2_)
                Width = Right - Left
                Height = Bottom - Top
                if (Width>0)and(Height>0):
                    square_intersection = Width * Height
                    if (square_intersection> k_intersecion * min(square, square_)):
                        results_list[j][5] = -1
    return results_list

@st.cache_resource()
def get_model(modeltype):
    modelpath='./models'
    if modeltype in model_names:
        from ultralytics import YOLO
        modelpath = os.path.join(modelpath, model_files[model_names.index(modeltype)])
        model = YOLO(modelpath)  # загрузка весов обученной нейронной сети 
        print(f"Модель '{modeltype}' успешно загружена ")
    else:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
    return model

#получение фрейма из потокового видео с камеры, его анализ и трансформация
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.class_names = class_names
        self.colors = colors
        if len(self.class_names)==0: self.class_names=model.names
    def get_preds(self, frame : np.ndarray) -> np.ndarray:
        return (self.model(frame)[0])
    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.get_preds(frame)
        results_list_g = [0] * len(self.class_names) #список обнаруженных классов
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score >= threshold:
                results_list_g[int(class_id)] += 1 
                color=(0,255,0)             
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                cv2.putText(frame, class_names[int(class_id)]+"-"+str(int(score*100))+"%", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_COMPLEX, 1.4, color, 3, cv2.LINE_AA)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Сохранение BytesIO на диск
def write_bytesio_to_file(filename, bytesio):
    """
    Записывает содержимое BytesIO в файл.
    Создает файл или перезаписывает файл, если его не существует. 
    """
    with open(filename, "wb") as outfile:
        # Копирование потока BytesIO в выходной файл
        outfile.write(bytesio.getbuffer())
#очистка временных файлов
def rm_temp():
    for filename in os.listdir(os.path.join('.','temp')):
        os.remove(os.path.join('./temp',filename))

@st.cache_data()
#классификация в соовтетствии с заданными параметрами
def classification(results, rows):
    count = 1 #счётчик дополнительных нарушений
    df = pd.DataFrame(columns=["наличие","нарушения"], index=list(set(rows))).fillna("")
    df_match = pd.read_csv("./uniformmatch.csv", sep=",", index_col=0).fillna(0)
    for classname in results.keys():
        if results[classname]>0:
            for col in df_match.columns:
                #если класса нет в списке классов, он лишний
                if (df_match.loc[classname, col]>0) and (df_match.loc['min', col]>0) and (col not in df.index):
                    if classname not in list(df['нарушения']):
                        df.loc[f'нарушение_{count}'] = ["", classname]
                        count+=1
                #если класс есть в списке, он не являетсся нарушением, количство соответствует
                elif (df_match.loc[classname, col]==1) and (col in df.index):
                    if (df_match.loc['min', col] <= results[classname] <= (df_match.loc['max', col])):
                        df.loc[col,'наличие'] = '+'
                    else:                                         
                        df.loc[col,'нарушения'] =  df.loc[col,'нарушения'] + "несоответствие количества\n"
                #если класс есть в списке и он является нарушением
                elif (df_match.loc[classname, col]==2) and (col in df.index):
                    df.loc[col,'наличие'] = '+'
                    df.loc[col,'нарушения'] = df.loc[col,'нарушения'] + "несоответствие требованиям\n"
    for row in rows:
        if df.loc[row,'наличие'] == "": df.loc[row,"нарушения"] = "отсутствует"
    return df

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

    """  
    Основные переменные используемые в main
    results_list - список с результатами классификации для всех загруженных изображений/видео, индекс соответствует id класса 
    result_list - список с результами классификации для одного изображения/видео, индекс соответствует id класса, 
                    значение равно максимальному количеству объктов данного класса на классифицируемом образе
    results_dict - словарь результатов классификации: key - имя класса, value - количество экземпляров класса 
    results_dict_g - глобальный словарь результатов классификации: key - имя класса, value - количество экземпляров класса 
                        исопользуется в классе Video Transformer для потокового преобразования видео
    сlass_names - словарь names цветов прямоугольников, ограничивающих классы {id_class: (B,G,R)} из файла  
    сolors - словарь color цветов прямоугольников, ограничивающих классы {id_class: (B,G,R)} из файла 
    model_names - список model_names из файла , содержащих наименования классификаторов, отображаемых на UI
    model_files - список model_files из файла , содержащих имена файлов моделей в папке models
    uniform - cловарь uniform определеяющий содержание проверяемого комплекта одежды {наименование_комплекта: [головной убро, туфли, .....]} из файла  
    """
def main():
    st.set_page_config(page_title="Внешний вид")
    st.title('Сервис оценки внешнего вида сотрудников',)
    rm_temp()
    global class_names, colors, model_names, model_files, uniform #словари наименования классов и цветов прямоугольников ограничиващих предметы по их id
    try:
        with open('config.yml',encoding="UTF-8") as fh:
            read_data = yaml.load(fh, Loader=yaml.FullLoader)
            path_dataset = read_data['path']
            path_imtrain = read_data['train']
            path_imval = read_data['val']
            class_names = read_data['names']
            colors = read_data['colors']
            model_names = read_data['model_names']
            model_files = read_data['model_files']
            uniform = read_data['uniform']
            uniform_consist=read_data['uniform_consist']
            print('Файл конфигурации config.yml успешно загружен')
    except:
        print('Отсутствует  или данные записаны в неправильном формате')
        path_dataset = {'path': '/home/bsa/Projects/UniformDetect/data'}
        path_imtrain = {'train': 'images/train'}
        path_imval = {'val': 'images/train'}
        class_names = {}
        colors={}
        model_names=[]
        model_files=[]
        uniform={}
        uniform_consist=[]
    
    st.sidebar.image("headimage.jpg")
    detect_mode = st.sidebar.radio("Тип входных данных",
                                   
                                    ('Изображение', 'Видеозапись', 'Камера'), index=0)
    
    type_uniform = st.sidebar.selectbox('Проверяемый комплект одежды', tuple(list(uniform.keys())) , index=0)
    #удаление или редактирование выбранного комплекта формы
    tab3, tab4, tab5 = st.sidebar.tabs(["📋", "📝","❌"])
    new_uniform = tab4.multiselect('Состав комплекта', list(uniform_consist),default=uniform[type_uniform])
    new_name = tab4.text_input('Наименование', value=type_uniform)
    if tab4.button('Сохранить'):
        if new_name != type_uniform: uniform[new_name] = new_uniform 
        else: uniform[type_uniform] = new_uniform 
        dump_all = [{'path': path_dataset}, {'train':path_imtrain}, {'val':path_imval}, {'names':class_names}, {'colors':colors},
                    {'model_names':model_names}, {'model_files':model_files}, {'uniform':uniform}, {'uniform_consist':uniform_consist}]
        try:
            with open('config.yml', 'w') as fw:
                yaml.dump_all(dump_all, fw, sort_keys=False, encoding='UTF-8', allow_unicode=True, Dumper=IndentDumper,\
                            explicit_end=False, explicit_start=False)
            fw = open("config.yml", "rt") 
            data = fw.read() 
            data = data.replace('---', '')
            data = data.replace('...', '')
            fw.close() 
            fw = open("config.yml", "wt") 
            fw.write(data)
            fw.close()
            print('Файл config.yml успешно обновлён' )
        except:
             print('Ошибка записи ')
    if tab5.button('Удалить комплект одежды'):
        del uniform[type_uniform]
        dump_all = [{'path': path_dataset}, {'train':path_imtrain}, {'val':path_imval}, {'names':class_names}, {'colors':colors},
                    {'model_names':model_names}, {'model_files':model_files}, {'uniform':uniform}, {'uniform_consist':uniform_consist}]
        try:
            with open('', 'w') as fw:
                yaml.dump_all(dump_all, fw, sort_keys=False, encoding='UTF-8', allow_unicode=True, Dumper=IndentDumper,\
                            explicit_end=False, explicit_start=False)
            fw = open("", "rt") 
            data = fw.read() 
            data = data.replace('---', '')
            data = data.replace('...', '')
            fw.close() 
            fw = open("", "wt") 
            fw.write(data)
            fw.close()
            print('Файл  успешно обновлён' )
        except:
             print('Ошибка записи ')       
             
    global threshold
    threshold = 0.01 * st.sidebar.slider("Порог обнаружения", min_value=0.0, max_value = 100.0, value=10.0) #порог обнаружения
    
    global model 
    if len(model_names)==0: 
        st.write('Алгоритмы распознавания не найдены. Используется алгоритм по умолчанию')
        model_type = "default"
    else:
        model_type = st.sidebar.selectbox('Алгоритм распознавания', tuple(model_names), index=0)

    model = get_model(model_type)
    if len(class_names)==0: class_names = model.names #если не заданы классы в файле 
    
    global result_list_g
    results_list_g = [0] * len(class_names) #список обнаруженных классов
    if detect_mode == 'Камера':
        if webrtc_streamer(key="Для проверки встаньте в объектив камеры", video_transformer_factory=VideoTransformer):
            results_dict = dict(zip(class_names.values(), results_list_g)) #словарь с результатами работы класс: количество
            st.dataframe(classification(results=results_dict, rows=uniform[type_uniform]), use_container_width=True)
    elif detect_mode == 'Видеозапись':
        uploaded_files = st.file_uploader("Выберите один или несколько видеофайлов", ['mp4','mov', 'avi'], accept_multiple_files=True)
        if st.button("Оценить внешний вид"):
            if uploaded_files:
                rm_temp()
                results_list = [0] * len(class_names) #список обнаруженных классов на всех видео
                tab1, tab2 = st.tabs(["🗃", "📈"])
                for uploaded_file in uploaded_files:
                    temp_file_to_save = os.path.join('./temp',uploaded_file.name)
                    temp_file_result_mp4 = os.path.join("./temp", '{}_out.{}'.format(uploaded_file.name[:uploaded_file.name.rfind(".")],uploaded_file.name[uploaded_file.name.rfind(".") + 1:])) 
                    temp_file_result_h264 = os.path.join("./temp", '{}_out_h264.{}'.format(uploaded_file.name[:uploaded_file.name.rfind(".")],uploaded_file.name[uploaded_file.name.rfind(".") + 1:])) 
                    # Сохранение загруженного видео на диск в качестве временного файла
                    write_bytesio_to_file(temp_file_to_save, uploaded_file)
                    cap = cv2.VideoCapture(temp_file_to_save)
                    # Определение параметров видео: ширина и высота фрейма, частота кадров
                    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    FPS = cap.get(cv2.CAP_PROP_FPS)
                    # подготовка к обработке и записи фреймов
                    fourcc_mp4 = cv2.VideoWriter_fourcc(*'MP4V') #используемый кодек
                    out_mp4 = cv2.VideoWriter(temp_file_result_mp4, fourcc_mp4, FPS, (W, H))
                    while True:
                        ret,frame = cap.read()
                        if not ret: break
                        results = model(frame)[0]
                        result_list = [0] * len(class_names) #список обнаруженных классов на одном изображении
                        for result in results.boxes.data.tolist():
                                    x1, y1, x2, y2, score, class_id = result
                                    if score > threshold:
                                        result_list[int(class_id)] += 1 #увеличиваем на 1 количество обнаруженных объектов данного класса
                                        color = (0,255,0)
                                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                                        cv2.putText(frame, class_names[int(class_id)]+"-"+str(int(score*100))+"%", (int(x1), int(y1 - 10)),
                                                    cv2.FONT_HERSHEY_COMPLEX, 1.4, color, 3, cv2.LINE_AA)
                        out_mp4.write(frame)
                    out_mp4.release()
                    cap.release()
                    cv2.destroyAllWindows() 
                    #Перекодировка видео из кодека mp4 в h264 используя ffmpeg
                    #Без перекодировки видео не проигрывается в браузере
                    #Требует установки ffmeg в ОС
                    #subprocess.call(args=f"ffmpeg -y -i {temp_file_result_mp4} -c:v libx264 {temp_file_result_h264}".split(" "))
                    tab1.video(temp_file_result_h264, start_time=0)
                    with open(temp_file_result_mp4, "rb") as file:
                        btn = tab1.download_button(
                        label="Загрузить",
                        data=file,
                        file_name='{}_out.{}'.format(uploaded_file.name[:uploaded_file.name.rfind(".")],uploaded_file.name[uploaded_file.name.rfind(".") + 1:]),
                        mime="video/mp4")
                    for i in range(len(result_list)):
                        if result_list[i] > results_list[i]: results_list[i] = result_list[i]
                    results_dict = dict(zip(class_names.values(), results_list)) #словарь с результатами работы класс: количество
    elif detect_mode == 'Изображение':
        uploaded_files = st.file_uploader("Выберите одно или несколько изображений", type=['jpg', 'jpeg', 'png','bmp','dib'], accept_multiple_files=True)
        if st.button("Оценить внешний вид"):
            if uploaded_files:
                rm_temp()
                tab1, tab2 = st.tabs(["🗃", "📈"])
                results_list = [0] * len(class_names) #список обнаруженных классов на всех изображениях
                for uploaded_file in uploaded_files:
                    temp_file_to_save = os.path.join('./temp',uploaded_file.name)
                    temp_file_result = os.path.join("./temp", '{}_out.{}'.format(uploaded_file.name[:uploaded_file.name.rfind(".")],uploaded_file.name[uploaded_file.name.rfind(".") + 1:])) 
                    # Сохранение загруженного видео на диск в качестве временного файла 
                    with open(temp_file_to_save, "wb") as f:
                        f.write(uploaded_file.getbuffer()) 
                    try:
                        img = cv2.imread(temp_file_to_save)
                    except:
                        print(f"file {temp_file_to_save} is not an image *.jpg, *.jpeg, *.png, *.bmp, *.dib")
                        continue 
                    H, W, _ =  img.shape
                    results = model(source=img)[0]
                    result_list = [0] * len(class_names) #список обнаруженных классов на одном изображении

                    # for result in results.boxes.data.tolist():
                    #     x1, y1, x2, y2, score, class_id = result 
                    #     if score > threshold:
                    #         result_list[int(class_id)] += 1 #увеличиваем на 1 количество обнаруженных объектов данного класса
                    #         color = (0,255,0)  
                    #         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                    #         cv2.putText(img, class_names[int(class_id)]+"-"+str(int(score*100))+"%", (int(x1), int(y1 - 10)),
                    #                     cv2.FONT_HERSHEY_COMPLEX, 1.4, color, 3, cv2.LINE_AA)
                    # cv2.imwrite(temp_file_result, img) 
                    # tab1.image(temp_file_result)

                    for result in remove_duplicate(results.boxes.data.tolist()):
                        x1, y1, x2, y2, score, class_id = result 
                        if (score > threshold) and (int(class_id) != -1):
                            result_list[int(class_id)] += 1 #увеличиваем на 1 количество обнаруженных объектов данного класса
                            filename = os.path.join('./temp','{}_{}.jpg'.format(int(class_id),result_list[int(class_id)]))
                            cv2.imwrite(filename, img[int(y1):int(y2), int(x1):int(x2)])
                            tab1.image(filename)
                            color = (0,255,0)  
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                            cv2.putText(img, class_names[int(class_id)]+"-"+str(int(score*100))+"%", (int(x1), int(y1 - 10)),
                                        cv2.FONT_HERSHEY_COMPLEX, 1.4, color, 3, cv2.LINE_AA)
                    cv2.imwrite(temp_file_result, img) 
                    tab1.image(temp_file_result)
                    
                
if __name__=="__main__":
    main()



