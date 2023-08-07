import numpy as np
import cv2
import os
import time
import mediapipe as mp

from ultralytics import YOLO 
import queue

import threading
from config import Q_SZ

from personDet import analysis_yolov8
import tools
from holisticDet import MediapipeProcess
import mediapipe_detection_image
# from PP_TSMv2_infer import PP_TSMv2_predict



class DealVideo():

    def __init__(self,video_file,video_save_file,person_model,mediapipe_model,pptsmv2_model):

        '''
        加载数据
        '''

        self.video_file = video_file
        self.video_save_file = video_save_file

        # 初始化模型

        self.person_model = person_model
        self.mediapipe_model = mediapipe_model
        self.pptsmv2_model = pptsmv2_model

        # 图片检测后队列
        self.videoQueue = queue.Queue(maxsize=Q_SZ)
        self.frameQueue = queue.Queue(maxsize=0)
        self.cutbboxQueue = queue.Queue(maxsize=0)
        self.videoframeQueue = queue.Queue(maxsize=0)
        self.videohandsQueue = queue.Queue(maxsize=0)
        self.videoheadQueue = queue.Queue(maxsize=0)
        self.videopersonQueue = queue.Queue(maxsize=0)

        #线程
        self.get_video_listThread = threading.Thread(target=self.get_video_list)
        self.get_video_frameThread = threading.Thread(target=self.get_video_frame)
        self.person_detThread = threading.Thread(target=self.person_det)
        self.write_videoThread = threading.Thread(target=self.write_video)
        self.select_video_frameThread = threading.Thread(target=self.select_video_frame)
        self.head_hands_detThread = threading.Thread(target=self.head_hands_det)
        

    def get_video_list(self):

        '''
        获取数据文件
        '''

        if os.path.isdir(self.video_file):

            video_ext = [".mp4", ".avi",".MP4"]
            for maindir, subdir, file_name_list in os.walk(self.video_file):
                for filename in file_name_list:
                    apath = os.path.join(maindir, filename)
                    ext = os.path.splitext(apath)[1]
                    if ext in video_ext:
                        self.videoQueue.put(apath)

        else:
            self.videoQueue.put(self.video_file)

    # def cut_video_seg(self):

    #     pass

    def get_video_frame(self):

        '''
        对视频进行分帧、每一帧都保存队列
        '''

        while True:

            if self.videoQueue.empty():

                time.sleep(2)
            
            else:
                
                try:

                    t1 = time.time()
                    video_path = self.videoQueue.get()  
                    

                    # video_basename = os.path.basename(video_path).split('.')[0]

                    cap = cv2.VideoCapture(video_path)

                    frame_list = []
                    count_fps = 0

                    while cap.isOpened():
                        success, frame = cap.read()
                        if not success:
                            print(video_path,"Ignoring empty camera frame.")
                            break
                        count_fps  += 1
                        # print('count_fps_read_video=',count_fps)

                        frame_dict = {'fps':count_fps,'frame':frame}
                        frame_list.append(frame_dict)
                        

                    video_dict = {'video_path':video_path,'frame_list':frame_list}
                
                    self.frameQueue.put(video_dict)

                    t2 = time.time()
                    # time.sleep(30)

                    print('视频读帧时间：',t2-t1,'总帧数：',len(frame_list))


                except Exception as e:
                    print(e)


    # def get_video_frame(self):

    #     '''
    #     对视频进行分帧、每一帧都保存队列
    #     '''
    #     while True:
    #         if self.videoQueue.empty():

    #             time.sleep(2)
    #         else:
                
    #             video_path = self.videoQueue.get()  
                    
    #             # video_basename = os.path.basename(video_path).split('.')[0]

    #             # 读取视频
    #             cap = cv2.VideoCapture(video_path)
    #             video_fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #             # print('video_fps:',video_fps)

    #             # 更新开始读帧
    #             start_frame = 0
    #             # frame_list = []
    #             count_fps = 0
    #             dert_fps = 80
    #             video_fps_1 = int(video_fps / dert_fps)

    #             # cap.set(propId = cv2.CAP_PROP_POS_FRAMES, value = start_frame)

    #             for s_frame_fps in range(video_fps_1):

    #                 frame_list = []

    #                 # print(s_frame_fps)

    #                 cap.set(propId = cv2.CAP_PROP_POS_FRAMES, value = start_frame)

    #                 while cap.isOpened():
    #                     success, frame = cap.read()
    #                     if not success:
    #                         print(video_path,"Ignoring empty camera frame.")
    #                         break
    #                     count_fps  += 1
    #                     # print('count_fps_read_video=',count_fps)
                                
    #                     frame_dict = {'fps':count_fps,'frame':frame}
    #                     frame_list.append(frame_dict)
    #                     # print('frame_list:',len(frame_list))

    #                     if len(frame_list) > dert_fps:
    #                         start_frame = count_fps
    #                         print('frame_dict_full',"start_frame_updata",start_frame)
    #                         break
    #                     # break

    #                 video_dict = {'video_path':video_path,'frame_list':frame_list}
    #                 # print('video_dict:',video_dict)
    #                 # print('frame_list_len',len(frame_list))
    #                 self.frameQueue.put(video_dict)
    #                 # time.sleep(2)




    def person_det(self):

        '''
        从队列中获取视频帧frame，进行第一步人员的检测
        '''
        # print('person_detshgshgsugh;')
        while True: 

            if self.videoframeQueue.empty():

                time.sleep(2)
            else:

                t0 = time.time()
                video_frame_dict = self.videoframeQueue.get()

                frame_list = video_frame_dict['frame_list']
                video_path = video_frame_dict['video_path']

                frame_result_contact = []
                # frame_list_seg = []

                for i in range(len(frame_list)):

                    if frame_list[i]["fps"] == i + 1:

                        # frame_list_seg.append(frame=frame_list[i]['frame'])       
                        # 
                        t1 = time.time()         

                        person_det = analysis_yolov8(frame=frame_list[i]['frame'],
                                                     model_coco=self.person_model,
                                                     confidence_set=0.5)
                        
                        t2 = time.time()
                        
                        # 当前帧检测的结果列表，只包含bboxlist
                        person_list = tools.get_dict_values(person_det)

                        if person_list:

                            # print('person_list:',person_list)

                            label_name = list(person_det[0].keys())[0]

                            update_frame_result_contact = self.get_cut_message(fps1=frame_list[i]["fps"],
                                                                                label_name = label_name,
                                                                                re_list=person_list,
                                                                                video_path=video_path,
                                                                                frame_result_contact=frame_result_contact,
                                                                                parameter_fps=40)
                            
                            frame_result_contact = update_frame_result_contact
                        
                        t3 = time.time()

                        print('yolov8推理时间：',t2-t1,'yolov8结果处理时间',t3-t2,'总时间：',t3-t0,'读取视频队列时间',t1-t0)
                        # print('frame_result_contact:',frame_result_contact)


    def head_hands_det(self):

        # print('head_hands_detaohgaogh')

        while True:

            if self.videopersonQueue.empty():

                time.sleep(5)
            else:

                t0 = time.time()
                person_frame_dict = self.videopersonQueue.get()

                # print('person_frame_dict')

                person_frame_list = person_frame_dict['frame_list']
                video_path = person_frame_dict['video_path']

                head_result_contact = []
                hands_result_contact = []   

                for i in range(len(person_frame_list)):

                    if person_frame_list[i]["fps"] == i + 1: 

                        image = person_frame_list[i]["frame"]

                        imgsize = image.shape

                        # hh_result_dict = mediapipe_detection_image.main(images=image,
                        #                                            face_b=50,
                        #                                            left_hand_b=7,
                        #                                            right_hand_b=7)
                        
                        # print("hh_result:",hh_result_dict)

                        # print(type(image))

                        t1 = time.time()
                        # 模型推理
                        hh_result = MediapipeProcess.mediapipe_det(image=image,
                                                                        holistic=self.mediapipe_model)
                        hh_result_dict = MediapipeProcess.get_analysis_result(image=image,results=hh_result)

                        t2 = time.time()
                        
                        # # 获得当前坐标列表
                        head_result = hh_result_dict['face_bbox']
                        head_result_1 = tools.select_list(head_result)
                        hands_result = hh_result_dict['hand_bbox']
                        hands_result_1 = tools.select_list(hands_result)

                        # print('head_result_1:',head_result_1)
                        # print('head_result_1:',hands_result_1)

                        # 获得左上和右下坐标


                        # # 统一修正坐标,分别对头和手进行分析
                        if head_result_1:
                            head_bbox_list = tools.para_list_correction(images_size=imgsize,
                                                                            bbox_list=head_result,
                                                                            dertpara=[])

                            # print('head_bbox_list:',head_bbox_list)

                            update_head_result_contact = self.get_cut_message(fps1=person_frame_list[i]["fps"],
                                                                                label_name = 'head',
                                                                                re_list=head_bbox_list,
                                                                                video_path=video_path,
                                                                                frame_result_contact=head_result_contact,
                                                                                parameter_fps=20)
                            head_result_contact = update_head_result_contact


                        if hands_result_1:

                            hands_bbox_list = tools.para_list_correction(images_size=imgsize,
                                                                            bbox_list=hands_result_1,
                                                                            dertpara=[]) 
                            
                            # print('hands_bbox_list:',hands_bbox_list)
                           
                            update_hands_result_contact = self.get_cut_message(fps1=person_frame_list[i]["fps"],
                                                                                label_name = 'hands',
                                                                                re_list=hands_bbox_list,
                                                                                video_path=video_path,
                                                                                frame_result_contact=hands_result_contact,
                                                                                parameter_fps=20)
                        
                            hands_result_contact = update_hands_result_contact
                        
                        t3 = time.time()

                        print('mediapipe推理时间',t2-t1,'mediapipe结果分析时间：',t3-t2,'总时间',t3-t0,'读取队列中图片的时间：',t1-t0)

                        # print("head_result_contact:",head_result_contact)
                        # print("hands_result_contact:",hands_result_contact)

    # def video_select_dect(self):

    #     while True:

    #         if ~self.videoheadQueue.empty():

    #             video_dict = self.videoheadQueue.get()

    #             video_path = video_dict["video_path"]

    #             result_list = PP_TSMv2_predict().predict(config='',
    #                                                      input_file=video_path,
    #                                                      batch_size='',
    #                                                      model_file='',
    #                                                      params_file='')
                
                  



    #     pass



    def get_cut_message(self,fps1,label_name,re_list,video_path,frame_result_contact,parameter_fps):


        if not frame_result_contact:

            bbox_list_all = tools.change_list_dict(fps1=fps1,re_list=re_list)

            frame_result_contact = bbox_list_all
            # print("frame_result_contact:",frame_result_contact)

        else:

            example_dict_list = frame_result_contact
            # print('example_dict_list:',example_dict_list)
            # print('re_list:',re_list)

            cut_list,example_lst,re_dict_lst = tools.analysis_re01_list(example_list=example_dict_list,
                                                                                result_list=re_list)
                            
            # print('cut_list:',cut_list)
            # print('example_sorted_lst:',example_lst)
            # print('re_dict_sorted_lst:',re_dict_lst)

                        
            # 有目标减少情况
            if example_lst:

                # 截图保存视频

                cut_dict = {'video_path':video_path,'label_name':label_name,"stop_fps":fps1,'bbox_list':example_lst}

                # 添加到新的队列
                self.cutbboxQueue.put(cut_dict)

                frame_result_contact = [item for item in frame_result_contact if item not in example_lst]
                                
            # 有新添加目标情况
            if re_dict_lst:

                # 对比示例列表更新
                update_list = tools.change_list_dict(fps1=fps1,re_list=re_dict_lst)

                frame_result_contact = frame_result_contact + update_list

            # 统计截止时间
            time_out_list = tools.statistics_fps(fps_now=fps1,re_list=frame_result_contact,parameter=parameter_fps)
                            

            if time_out_list:

                # 裁剪保存视频
                # bbox_list = Process_tools.change_dict_list(time_out_list)
 
                cut_dict = {'video_path':video_path,'label_name':label_name,"stop_fps":fps1,'bbox_list':time_out_list}

                # 添加到新的队列
                self.cutbboxQueue.put(cut_dict)

                # 对比示例列表更新
                frame_result_contact = [item for item in frame_result_contact if item not in time_out_list]

            # print('frame_result_contact:',frame_result_contact)
        
        return frame_result_contact


    def write_video(self): 

        # print('write_videoafagragr')

        '''  
        保存成视频
        '''

        while True:

            if self.cutbboxQueue.empty():

                time.sleep(5)

            else:

                t1 = time.time()

                video_frame_dict = self.cutbboxQueue.get()

                # print('video_frame_dict:',video_frame_dict)

                # 视频路径
                video_path = video_frame_dict['video_path']
                video_basename = os.path.basename(video_path).split('.')[0]
                file_name = video_frame_dict['label_name']
                # video_name_save = os.path.join(self.video_save_file, video_basename)

                # 原视频帧率和尺寸
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 获得起始帧
                stop_fps = video_frame_dict['stop_fps']

                # 裁剪信息
                result_list = video_frame_dict['bbox_list']

                for i,bbox_dict in enumerate(result_list):

                    start_fps = bbox_dict['fps']
                    bbox_list = bbox_dict['result']

                    w = int(bbox_list[2]) - int(bbox_list[0])
                    h = int(bbox_list[3]) - int(bbox_list[1])

                    size = (w,h)

                    # 根据标签保存不同视频分类
                    video_name_save = video_basename + '_' + str(start_fps)  + '_' + str(stop_fps) + '_'  + str(i) + '.avi'
                    video_save_file = self.video_save_file + '/' + file_name
                    os.makedirs(video_save_file, exist_ok=True)
                    video_save_path = os.path.join(video_save_file, video_name_save)

                    videoWriter =cv2.VideoWriter(video_save_path,cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)

                    tools.save_seg_video(video_name=cap,
                                                    frameToStart=start_fps,
                                                    frametoStop=stop_fps,
                                                    videoWriter=videoWriter,
                                                    bbox=bbox_list)
                        
                    videoWriter.release()

                    self.videoQueue.put(video_save_path)

                    t2 = time.time()

                cap.release()

                t3 = time.time()

                print('保存一个位置的时间：',t2-t1,'保存一个视频片段中多有目标的时间：',t3-t1)


    def select_video_frame(self):

        # print('select_video_frameaoghroaghouthg')

        while True:

            if self.frameQueue.empty():

                time.sleep(1)
            else:

                t1 = time.time()

                video_dict = self.frameQueue.get()
                video_path = video_dict["video_path"]   
                directory = os.path.dirname(video_path)
                labels = directory.split('/')[-1]

                # print('labels:',labels)

                if labels == 'person':

                    self.videopersonQueue.put(video_dict)

                if labels == 'head':

                    # print('youshou')

                    self.videoheadQueue.put(video_dict)
                
                if labels == 'hands':

                    # print('youshou')

                    self.videoheadQueue.put(video_dict)

                else:

                    self.videoframeQueue.put(video_dict)

                # print('end_select')
                t2 = time.time()

                print('挑选保存数组的时间：',t2-t1)


    def run(self):

        self.get_video_listThread.start()
        self.get_video_frameThread.start()
        self.person_detThread.start()
        self.write_videoThread.start()
        self.select_video_frameThread.start()
        self.head_hands_detThread.start()



if __name__ == '__main__':  
  
    t1 = time.time()

    video = "E:/Bank_files/Bank_02/dataset/video_test/1min/0711-7_4.avi"
    video_save = 'test_video'

    person_model = YOLO("model_file/yolov8n.pt")

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

    # get_seg_video(video_file=video,video_save_path=video_save,dertTime=dertTime)
    
    deal = DealVideo(video_file=video,video_save_file=video_save,person_model=person_model,mediapipe_model=holistic,pptsmv2_model='model_file/yolov8x_person.pt')
    deal.run()

    t2 = time.time()

    print('总时间：',t2-t1)

    








