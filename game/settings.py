class Settings(object):
    '''存储 外星人入侵 的所有设置的类'''
    def __init__(self):
        '''store the game's setting存储游戏中所有设置的类'''
        #屏幕设置
        self.screen_width=800
        self.screen_height=600
        self.bg_color=(230,230,230)
        #飞船的设置
        self.ship_speed_factor=1.5
        # 设置飞船限制的个数为3，即允许3次复活
        self.ship_limit=3

        #子弹类
        self.bullet_speed_factor=1
        self.bullet_width=3
        self.bullet_height=15
        self.bullet_color=(60,60,60)
        self.bullets_allowed=3
        #外星人设置
        #外星人速度
        self.alien_speed_factor=1
        #外星人装到屏幕边缘时，外星人群向下移动的速度
        self.alien_drop_speed=50
        #fleet_direction为1代表右向，-1代表左向
        self.fleet_direction=1
