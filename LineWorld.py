# 直線のセカイを構築
class LineWorld:
    def __init__(self,length):
        # 長さ
        self.length = length
        # 位置
        self.pos = 0
        # 範囲
        self.goal = self.length
        self.end = -self.length


    def reset(self):
        # やり直し
        self.pos = 0
        return [self.pos / self.goal]  
    def step(self,action):
        # action => 1 or -1
        move = -1 if action == 0 else 1
        
        self.pos += move
        
        self.pos = min(self.goal ,max(self.end, self.pos))  # 端で止まる
        
        done = False
        reward = 0

        if self.pos == self.length:
            reward = 10
            done = True
        elif move == 1:
            reward = 1
        elif move == -1:
            reward = -5

        return [self.pos / self.goal], reward, done