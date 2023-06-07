import cv2
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import os

import cv2
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions import hands
import time
import os
import queue
import threading


from pipeline_getResult import atm_det


class analysis_result():

    def __init__(self,imgPath,savePath,modellist):

        self.imgPath = imgPath
        self.savePath = savePath

        self.imgList = os.listdir(self.imgPath)

        #定义加载好的模型
        self.model_person = modellist[0]
        self.model_pp_hand = modellist[1]
        self.model_blue = modellist[2]
        self.model_screen = modellist[3]

        self.media_hands = modellist[4]

        # 队列
        self.imgQueue1 = queue.Queue(maxsize=len(self.imgList))
        self.imgQueue2 = queue.Queue(maxsize=len(self.imgList))
        self.imgQueue3 = queue.Queue(maxsize=len(self.imgList))
        self.imgQueue4 = queue.Queue(maxsize=len(self.imgList))
        self.imgQueue5 = queue.Queue(maxsize=len(self.imgList))
        self.imgQueue6 = queue.Queue(maxsize=len(self.imgList))

        #线程
        # self.get_imgThread = threading.Thread(target=self.get_img)
        self.analysis_screen_blueThread = threading.Thread(target=self.analysis_screen_blue)
        self.analysis_hands_blueThread = threading.Thread(target=self.analysis_hands_blue)
        self.analysis_one_buttonThread = threading.Thread(target=self.analysis_one_button)
        # self.get_pph_resultThread = threading.Thread(target=self.get_pph_result)
        # self.analysis_handThread = threading.Thread(target=self.analysis_hand)
        self.draw_imagesThread = threading.Thread(target=self.draw_images)
        # self.analysis_hand_blueThread = threading.Thread(target=self.analysis_hand_blue)
        # self.get_screen_resultThread = threading.Thread(target=self.get_screen_result)
   
    

    # 分析按钮与屏幕的关系，给每张图片输出响应的结果
    def analysis_screen_blue(self):

        # 暂存信息列表
        screen_none_list = []
        
        
        num = 0
        for img in self.imgList:

            screen_list = []

            num = num + 1
            imgpath = os.path.join(self.imgPath,img)

            print(num,'imgpath_begin:',imgpath)

            print('--------------began the img ',imgpath,'-------------------------------------------------------------------------------------------')

            screen_result = atm_det.get_screen_result(imgpath,self.model_screen)

            # 检测到屏幕
            if screen_result:

                    screen_id = list(screen_result[0].keys())[0]

                    # 检测按钮
                    blue_result = atm_det.get_blue_result(imgpath,self.model_blue)

                    blue_list = []

                    if blue_result:

                        num = 0

                        for num  in range(len(blue_result)):

                            blueDict = {num:blue_result[num]}

                            blue_list.append(blueDict)

                    # 没检测到按钮
                    else:
                        blue_list.append('None')
                    
                    screen_none_list.clear()
                    screen_copy = {"screen_id":screen_id,'blue_list':blue_list}
                    screen_none_list.append(screen_copy)

                    img_scr_dict = {'imgpath':imgpath,"screen_id":screen_id,'blue_list':blue_list}
                    screen_list.append(img_scr_dict)

            # 没检测到屏幕
            else:
                    # 检测按钮
                    blue_result = atm_det.get_blue_result(imgpath,self.model_blue)
                    # print('screen_none_list:',screen_none_list)

                    if screen_none_list:

                        blue_copy = screen_none_list[0]['blue_list']
                        screen_id_copy = screen_none_list[0]['screen_id']
                    
                        blue_list = []
                        # 上一张图有屏幕没按钮
                        if blue_copy[0] == 'None':

                            if blue_result:

                                num = 0

                                for num  in range(len(blue_result)):

                                    blueDict = {num:blue_result[num]}
                                    # print('updata:',blueDict)

                                    blue_list.append(blueDict)

                            else:
                                blue_list.append('None')
                                continue

                                
                        # 上一张图检测按钮比这张少
                        elif len(blue_result) >= len(blue_copy):

                            for num  in range(len(blue_result)):

                                    blueDict = {num:blue_result[num]}

                                    blue_list.append(blueDict)
                                    # print('D:',blueDict)



                        # 上一张图检测到的按钮比这张图多
                        elif len(blue_result) < len(blue_copy):
                            
                            if blue_result:

                                blue_list = blue_copy
                                # print('blue_copy:',blue_copy)

                            else:
                                blue_list.append('None')
                                continue  

                        # print('blue_list',blue_list)

                        img_scr_dict = {'imgpath':imgpath,"screen_id":screen_id_copy,'blue_list':blue_list}
                        screen_list.append(img_scr_dict)
            print('screen_list:',screen_list)

            self.imgQueue2.put(screen_list)

    # 通过按钮列表，选择指定位置按钮为确认键
    def analysis_one_button(self):
        # while True:

            if ~self.imgQueue2.empty():

                img_screen_list = self.imgQueue2.get()

                # print(img_screen_list)
                print('screen_list_2:',img_screen_list)

                if img_screen_list:

                    img_screen_dict = img_screen_list[0]

                    # screen_id = img_screen_dict['screen_id']

                    # screen_labels = [{1:['Bank_Card','ID_Card','Successful_Transaction_2']},
                    #                  {2:['Information_Writing','Statement','Password_Reset','Enter_Password_1','Enter_Password_2','Successful_Transaction_1','continue_Y_Or_N']},
                    #                  {3:['Face_Verification','Information_Confirmation','Sign']},
                    #                  {4:['Select_Service_Item ']}]
                    
                    # for i in range(4):

                    #     if screen_id in list(screen_labels[i].values())[0]:

                    #         point_num = int(list(screen_labels[i].keys())[0]) - 1

                    # 取最后一位位置为确认键、可修改
                    blue_list = img_screen_dict['blue_list'][0]

                    img_screen_dict.update({'blue_list':blue_list})

                    print('img_screen_dict_2:',img_screen_dict)

                    self.imgQueue3.put(img_screen_dict)
    
    # 结合选择后的坐标，输出按到确认键的手指的坐标信息和确认键的信息
    def analysis_hands_blue(self):

        # while True:

            if ~self.imgQueue3.empty():

                img_scr_dict = self.imgQueue3.get()

                imgpath = img_scr_dict['imgpath']
                img_button_area = list(img_scr_dict['blue_list'].values())[0]

                hands_result = atm_det.get_hand_landmarker(imgpath=imgpath,
                                                               model_person=self.model_person,
                                                               model_hands=self.media_hands)
                
                # print('hands_result:',hands_result)
                hand_button_list = []
                hand_button_list_pass = []
                button_list_pass = []
                if  hands_result:   
                
                    for hands_num in hands_result:

                        if hands_num:

                            for hands_axis in hands_num:

                                if img_button_area[0]  <= hands_axis[0] <= img_button_area[2] and img_button_area[1] <= hands_axis[1] <= img_button_area[3]:

                                    hand_button = {imgpath:{'hand':hands_axis,'button':img_button_area}}
                                    hand_button_list.append(hand_button)

                                else:

                                    hand_button_pass = {imgpath:{'hand':hands_num,'button':img_button_area}}
                                    hand_button_list_pass.append(hand_button_pass)
                        else:
                            button_list_pass = {imgpath:{'button':img_button_area}}
                else:

                    button_list_pass = {imgpath:{'button':img_button_area}}
                
                print('hand_button_list:',hand_button_list)
                print('hand_button_list_pass:',hand_button_list_pass)
                print('button_list_pass:',button_list_pass)

                all_Dict = [{'hand_button_list':hand_button_list},{'hand_button_list_pass':hand_button_list_pass},{'button_list_pass':button_list_pass}]

                print('all_Dict:',all_Dict)
                
                self.imgQueue4.put(all_Dict)



    def draw_images(self):
        num = 0
        while True:
            if ~self.imgQueue2.empty():
                
                img_screen_list = self.imgQueue4.get()
                num = num + 1
                print(num,'img_screen_list:',img_screen_list)


   
    # def draw_images(self):
        
    #     num = 0
    #     while True:
    #         if ~self.imgQueue4.empty():

    #             all_Dict = self.imgQueue4.get()

    #             if all_Dict:

    #                 for dicts1 in all_Dict:

    #                     if list(dicts1.keys())[0] == 'hand_button_list':

    #                         hand_button_list = list(dicts1.values())[0]
    #                         # print(hand_button_list)

    #                         if hand_button_list:

    #                             hand_button = hand_button_list[0]
    #                             print('hand_button:',hand_button)

    #                             # print(hand_button)

    #                             images = list(hand_button.keys())[0]
    #                             # print(images)

    #                             imgname = os.path.basename(images)

    #                             img = cv2.imread(images)

    #                             point_hand = list(hand_button.values())[0]['hand']

    #                             rec_blue = list(hand_button.values())[0]['button']


    #                             # 画手指点
    #                             cv2.circle(img, point_hand, 1,(0,0,255), 2)

                                
    #                             # 画矩形框
    #                             cv2.rectangle(img, (int(rec_blue[0]), int(rec_blue[1])),(int(rec_blue[2]), int(rec_blue[3])), (0, 0, 255), 1)

    #                             cv2.imwrite(os.path.join(self.savePath,imgname),img)
    #                         else:
    #                             print('hand_button_list_none')

    #                     elif list(dicts1.keys())[0] == 'hand_button_list_pass':

    #                         hand_button_list = list(dicts1.values())[0]
    #                         if hand_button_list:

    #                             hand_button = hand_button_list[0]

    #                             images = list(hand_button.keys())[0]

    #                             imgname = os.path.basename(images)

    #                             img = cv2.imread(images)

    #                             point_hand_list = list(hand_button.values())[0]['hand']

    #                             rec_blue = list(hand_button.values())[0]['button']

    #                             for point_hand in point_hand_list:
                                    
    #                                 # 画手指点
    #                                 cv2.circle(img, point_hand, 1,(0,255,0), 2)
                                
    #                             # 画矩形框
    #                             cv2.rectangle(img, (int(rec_blue[0]), int(rec_blue[1])),(int(rec_blue[2]), int(rec_blue[3])), (0, 255, 0), 1)

    #                             cv2.imwrite(os.path.join(self.savePath,imgname),img)
    #                         else:
    #                             print('hand_button_list_none')
    #                     else:

    #                         hand_button_list = list(dicts1.values())[0]

    #                         if hand_button_list:
    #                             # print('hand_button_list:',hand_button_list)

    #                             hand_button = hand_button_list

    #                             images = list(hand_button.keys())[0]

    #                             imgname = os.path.basename(images)

    #                             img = cv2.imread(images)


    #                             rec_blue = list(hand_button.values())[0]['button']
                                
    #                             # 画矩形框
    #                             cv2.rectangle(img, (int(rec_blue[0]), int(rec_blue[1])),(int(rec_blue[2]), int(rec_blue[3])), (255, 0, 0), 1)

    #                             cv2.imwrite(os.path.join(self.savePath,imgname),img)
    #                         else:
    #                             print('none')
    #             else:
    #                 print("none")

    #             num = num + 1
    #             print(num)


    
    def run(self):

        self.analysis_one_buttonThread.start()
        self.analysis_screen_blueThread.start()
        self.analysis_hands_blueThread.start() 
        self.draw_imagesThread.start()       


if __name__ == '__main__':
    

    model_person = YOLO("model_files/bk1.pt")
    model_pp_hand = YOLO("model_files/best_pph.pt")
    model_blue = YOLO("model_files/best_butten.pt")
    model_screen = YOLO("model_files/best_screen.pt")


    media_hands = hands.Hands(
          static_image_mode=True,
          max_num_hands=4,
          min_detection_confidence=0.1,
          min_tracking_confidence=0.1)
    
    modelList = [model_person,model_pp_hand,model_blue,model_screen,media_hands]

    # imgPath= 'E:/BANK_XZ/data_file'

    # imgList = os.listdir(imgPath)

    q = analysis_result(imgPath='E:/BANK_XZ/data_file_1',
            savePath='E:/Images Data/XZ/test_for_others/hands_button',
            modellist=modelList)
    
    q.run()



    # for img in imgList:

    #     imgpath = os.path.join(imgPath,img)

    #     handsre = atm_det.get_hand_landmarker(imgpath)
