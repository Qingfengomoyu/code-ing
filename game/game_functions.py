import sys
import os
import pygame
from bullet import  Bullet
from Alien import Alien
from time import sleep
def check_events(ai_settings,screen,ship,bullets):
    '''响应按键和鼠标事件'''
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key==pygame.K_RIGHT:
                #向右移动飞船
                ship.moving_right=True
            elif event.key==pygame.K_LEFT:
                ship.moving_left=True
            elif event.key==pygame.K_SPACE:
                #创建一个子弹，并加入到编组Bullets中
                # 限制子弹数最多为3
                if len(bullets)<ai_settings.bullets_allowed:
                    new_bullet=Bullet(ai_settings,screen,ship)
                    bullets.add(new_bullet)
            elif event.key==pygame.K_q:
                sys.exit()

        elif event.type==pygame.KEYUP:
            if event.key==pygame.K_RIGHT:
                ship.moving_right=False
            elif event.key==pygame.K_LEFT:
                ship.moving_left=False

def update_screen(ai_settings,screen,ship,aliens,bullets):
    screen.fill(ai_settings.bg_color)
    for bullet in bullets:
        bullet.draw_bullet()
    #显示飞船
    ship.blitme()
    #显示外星人群组
    aliens.draw(screen)
    # 让最近绘制的屏幕可见
    pygame.display.flip()
    #在飞船和外星人后面重绘所有子弹


def create_fleet(ai_settings,screen,aliens,ship):
    '''创建外星人群'''
    #创建一个外星人，并计算一行容纳多少人
    #外星人间距为外星人宽度
    alien=Alien(ai_settings,screen)
    alien_width=alien.rect.width
    avaiable_space_x=ai_settings.screen_width-2*alien_width
    number_aliens_x=int(avaiable_space_x/(2*alien_width))

    #计算屏幕中竖直空间，并计算可容纳多少列外星人
    alien_height=alien.rect.height
    ship_height=ship.rect.height
    avaiable_space_y=ai_settings.screen_height-(3*alien_height)-ship_height
    number_rows=int(avaiable_space_y/(2*alien_height))

    for row_number in range(number_rows):
        #创建第一行外星人
        for alien_number in range(number_aliens_x):
            alien=Alien(ai_settings,screen)
            alien.x=alien_width+2*alien_width*alien_number
            alien.rect.x = alien.x
            # 以下两行代码，设置新创建外星人的高度
            alien.y=alien_height+2*alien_height*row_number
            alien.rect.y=alien.y
            #........................
            aliens.add(alien)


def update_aliens(ai_settings,aliens):
    '''更新外星人群中所有外星人的位置'''
    check_fleet_edges(ai_settings,aliens)
    aliens.update()

def check_fleet_edges(ai_settings,aliens):
    '''有外星人到达边缘时采取相应的措施'''
    for alien in aliens.sprites():
        if alien.check_edge():
            change_fleet_direction(ai_settings,aliens)
            break
def change_fleet_direction(ai_settings,aliens):
    '''将整体外星人去下移，并改变他们的方向'''
    for alien in aliens.sprites():
        alien.rect.y+=ai_settings.alien_drop_speed
    ai_settings.fleet_direction*=-1

def ship_hit(ai_settings,stats,screen,ship,aliens,bullets):
    '''响应外星人撞到飞船'''
    if stats.ships_left>0:
        #将ship_left减1
        stats.ships_left-=1
        print(stats.ships_left)

        #清空外星人列表和子弹列表

        aliens.empty()
        bullets.empty()
        #创建一群新的外星人，并置于屏幕底部中央
        create_fleet(ai_settings,screen,aliens,ship)
        ship.center_ship()
        # 暂停0.5秒
        sleep(0.5)
    else:
        stats.game_active=False

def check_aliens_bottom(ai_settings,stats,screen,ship,aliens,bullets):
    '''检测是否有外星人到达了屏幕底端'''
    screen_rect=screen.get_rect()
    for alien in aliens.sprites():
        if alien.rect.bottom>=screen_rect.bottom:
            #像飞船一样处理
            ship_hit(ai_settings,stats,screen,ship,aliens,bullets)
            break