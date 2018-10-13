#coding=utf-8
import pygame
from pygame.locals import *
import sys
import time
import random


class Base(object):
    def __init__(self,screen_temp,x,y,image_name):
        self.x=x
        self.y=y
        self.bullet_list=[]
        self.screen=screen_temp
        self.image=pygame.image.load(image_name)
class BasePlane(Base):
    def __init__(self,screen_temp,x,y,image_name):
        Base.__init__(self,screen_temp,x,y,image_name)
    def display(self):
        self.screen.blit(self.image,(self.x,self.y))
        needDelItemList=[]
        for bullet in self.bullet_list:
         #   bullet.display()
          #  bullet.move()
        #    if bullet.judge():
         #       self.bullet_list.remove(bullet)
            if bullet.judge():
                needDelItemList.append(bullet)
        for bullet in needDelItemList:
            self.bullet_list.remove(bullet)
        for bullet in self.bullet_list:
            bullet.display()
            bullet.move()
        
class HeroPlane(BasePlane):
    def __init__(self,screen_temp):
        BasePlane.__init__(self,screen_temp,210,500,"./feiji/hero1.png")
        self.boom_list=[]
        self.hit= False
        self.image_index=0
        self.image_num=0
        self.create_image()

    #重写display方法
    def display(self,enemy_temp):
        for bullet in enemy_temp.bullet_list:
            if bullet.x>self.x and bullet.x<self.x+100:
                if bullet.y>self.y and bullet.y<self.y+124:
                    self.hit=True
        if self.hit == True:
            self.screen.blit(self.boom_list[self.image_index],(self.x,self.y))
            self.image_num+=1
            if self.image_num ==10:
                self.image_num=0
                self.image_index+=1
            if self.image_index>3:
                time.sleep(1)
                exit()
        else:
            self.screen.blit(self.image,(self.x,self.y))
            for bullet in self.bullet_list:
                bullet.display()
                bullet.move()
            
    def create_image(self):
        self.boom_list.append(pygame.image.load("./feiji/hero_blowup_n1.png"))
        self.boom_list.append(pygame.image.load("./feiji/hero_blowup_n2.png"))
        self.boom_list.append(pygame.image.load("./feiji/hero_blowup_n3.png"))
        self.boom_list.append(pygame.image.load("./feiji/hero_blowup_n4.png"))
    def move_left(self):
        self.x-=5
    def move_right(self):
        self.x+=5
    def fire(self):
        self.bullet_list.append(Bullet(self.screen,self.x,self.y))
    def bomb(self):
        self.hit=True
class EnemyPlane(BasePlane):
    def __init__(self,screen_temp):
        BasePlane.__init__(self,screen_temp,0,0,"./feiji/enemy0.png")
        self.direction="right"
        self.boom_list=[]
        self.hit= False
        self.image_index=0
        self.image_num=0
        self.create_image()
        
    def display(self,hero_temp):

        for bullet in hero_temp.bullet_list:
            if bullet.x>self.x and bullet.x<self.x+51:
                if bullet.y>self.y and bullet.y<self.y+39:
                    self.hit=True
        if self.hit == True:
            self.screen.blit(self.boom_list[self.image_index],(self.x,self.y))
            self.image_num+=1
            if self.image_num ==10:
                self.image_num=0
                self.image_index+=1
            if self.image_index>3:
                time.sleep(1)
                exit()
        else:
            self.screen.blit(self.image,(self.x,self.y))
            for bullet in self.bullet_list:
                bullet.display()
                bullet.move()
    def create_image(self):
        self.boom_list.append(pygame.image.load("./feiji/enemy0_down1.png"))
        self.boom_list.append(pygame.image.load("./feiji/enemy0_down2.png"))
        self.boom_list.append(pygame.image.load("./feiji/enemy0_down3.png"))
        self.boom_list.append(pygame.image.load("./feiji/enemy0_down4.png"))

    def fire(self):
        random_num=random.randint(0,100)
        if random_num==1 or random_num==72:
            self.bullet_list.append(EnemyBullet(self.screen,self.x,self.y))
    def move(self):
        if self.hit==False:
            if self.direction == 'right':
                self.x+=5
            elif self.direction == 'left':
                self.x-=5
            if self.x>480-50:
                self.direction="left"
            elif self.x<0:
                self.direction="right"
        else:
            pass
class BaseBullet(Base):
    def __init__(self,screen_temp,x,y,image_name):
        Base.__init__(self,screen_temp,x,y,image_name)
        
    def display(self):
        self.screen.blit(self.image,(self.x,self.y))

        
class Bullet(BaseBullet):
    def __init__(self,screen_temp,x,y):
        BaseBullet.__init__(self,screen_temp,x+40,y-20,"./feiji/bullet.png")
    
  
    def move(self):
        self.y-=20
    def judge(self):
        if self.y<0:
            return True
        else :
            return False
        
class EnemyBullet(BaseBullet):
    def __init__(self,screen_temp,x,y):
        BasePlane.__init__(self,screen_temp,x+25,y+40,"./feiji/bullet1.png")
        

    def move(self):
        self.y+=20
    def judge(self):
        if self.y>500:
            return True
        else :
            return False
def key_control(hero_temp):
    for event in pygame.event.get():
            #判断是否点击了退出按钮
        if event.type == QUIT:
            pygame.quit()
            print("exit")  
            sys.exit()
        elif event.type == KEYDOWN:
                #检测按键是否是a或者left
            if event.key == K_a or event.key == K_LEFT:
                 print('left')
                 hero_temp.move_left()
                #检测按键是否是d或者right
            elif event.key == K_d or event.key == K_RIGHT:
                 print('right')
                 hero_temp.move_right()
                #检测按键是否是空格键
            elif event.key == K_SPACE:
                 print('space')
                 hero_temp.fire()
            elif event.key ==K_b:
                 print('b')
                 hero_temp.bomb()




def main():
    #创建一个窗口，用来显示内容
    screen= pygame.display.set_mode((480,852),0,32)

    #创建一个和窗口大小的图片，用来充当背景
    background=pygame.image.load("./feiji/background.png")

    #把背景图片放到窗口中显示
    
    #创建一个飞机图片
    hero=HeroPlane(screen)
    enemy=EnemyPlane(screen)
    while True:
    #设定需要显示的背景图
        screen.blit(background,(0,0))
        hero.display(enemy)
        
        enemy.display(hero)
        enemy.move()
        enemy.fire()
        
        key_control(hero)
               
    #更新需要显示的内容
        pygame.display.update()
        time.sleep(0.01)




        
if __name__=='__main__':
    main()
    
