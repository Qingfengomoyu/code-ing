import sys
import os
import pygame
from settings import Settings
from ship import Ship
import game_functions as gf
from pygame.sprite import Group
from Alien import Alien
from game_stats import GameStats
def run_game():
    #init the pygame
    pygame.init()
    ai_settings=Settings()
    screen=pygame.display.set_mode((ai_settings.screen_width,ai_settings.screen_height))
    pygame.display.set_caption('Alien Insavion')
#设置背景色
    ship=Ship(ai_settings,screen)
    bullets=Group()
    aliens=Group()
    # 创建外星人群，传入ai_settings，screen，ship是为了计算能容纳多少行，列外星人
    gf.create_fleet(ai_settings,screen,aliens,ship)
    #创建一个外星人
    # alien=Alien(ai_settings,screen)
    #创建一个用于存储游戏统计信息的实例
    stats=GameStats(ai_settings)
    #开始游戏的主循环：
    while True:
        #监视键盘和鼠标事件
        gf.check_events(ai_settings,screen,ship,bullets)
        # 更改了语句，原文以状态变量stats.game_active来判断，此处以复活次数是否减到0来判断
        if stats.ships_left:

            #更新飞船位置
            ship.update()
            #更新外星人位置
            gf.update_aliens(ai_settings,aliens)



            #更新子弹的位置
            bullets.update()
            #删除已经消失的子弹
            for bullet in bullets.copy():
                if bullet.rect.bottom<=0:
                    bullets.remove(bullet)
            #检查是否有子弹击中了外星人
            #如果是这样，就删除相应的子弹和外星人,函数遍历aliens，当子弹和外星人的rect重叠时，返回一字典
            # 两个实参True告诉pygame删除发生碰撞时的子弹和外星人
            collisions=pygame.sprite.groupcollide(bullets,aliens,True,True)
            if len(aliens)==0:
                bullets.empty()
                gf.create_fleet(ai_settings, screen, aliens, ship)

            #检测外星人和飞船之间的碰撞
            if pygame.sprite.spritecollideany(ship,aliens):
                print('Ship hit!!!')
                gf.ship_hit(ai_settings,stats,screen,ship,aliens,bullets)

            if stats.ships_left>0:
                gf.check_aliens_bottom(ai_settings, stats, screen, ship, aliens, bullets)
        #每次循环都重绘屏幕
        gf.update_screen(ai_settings,screen,ship,aliens,bullets)


run_game()