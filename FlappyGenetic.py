import pygame
import random
import time
import numpy as np

pygame.init()

class NeuralNetwork:
    def __init__(self,x,y,z, weights1, weights2,
                 bias1, bias2):
        self.X = x
        self.Y = y
        self.Z = z
        self.weights1 = weights1
        self.weights2 = weights2
        self.bias1 = bias1
        self.bias2 = bias2


    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def predict(self):
        self.Data = np.array([self.X,self.Y,self.Z])
        self.layer2 = self.sigmoid(np.dot(self.Data, self.weights1) + self.bias1)
        self.layer3 = self.sigmoid(np.dot(self.layer2, self.weights2) + self.bias2)
        return self.layer3

class Bird:
    def __init__(self,Bird_Image, Bird_Mask, weights1 = np.random.rand(3,7) - 0.5,weights2 = np.random.rand(7,1) - 0.5,
                 bias1 = np.random.rand(1, 7) - 0.5, bias2 = np.random.rand(1, 1) - 0.5):
        self.Bird_X = 50
        self.Bird_Y = 50
        self.Gravity = 0.75
        self.acc = 0.075
        self.Bird_Image = Bird_Image
        self.Bird_Mask = Bird_Mask
        self.weight_1 = weights1
        self.weight_2 = weights2
        self.bias_1 = bias1
        self.bias_2 = bias2

        self.Score = 0 # Fitness of bird
        self.Pipe_Y = 0
        self.Pipe_distance = 0

        self.Bird_Distance_With_Pipe = 0
        self.Pipe_Height = 0

        self.Bird_Network = NeuralNetwork(self.Bird_Y, self.Bird_Distance_With_Pipe,
                                          self.Pipe_Height, self.weight_1, self.weight_2,
                                          self.bias_1, self.bias_2)


    def Draw_Bird(self,window):
        window.blit(self.Bird_Image,(int(self.Bird_X), int(self.Bird_Y)))

    def Update_NN(self):
        self.Bird_Network.X = self.Bird_Y
        self.Bird_Network.Y = self.Bird_Distance_With_Pipe
        self.Bird_Network.Z = self.Pipe_Height

    def Bird_Loop(self):
        #print(f"Skor = {self.Score}, Sonraki Pipe = {self.Bird_Distance_With_Pipe}")
        if self.Bird_Y < 0:
            return "Died"
        elif self.Bird_Y < 380:
            self.Bird_Y += self.Gravity
            self.Gravity += self.acc
        else:
            return "Died"



    def Bird_Jump(self):
        if self.Bird_Network.predict() < 0.5:
            self.Gravity = -2


class Pipe:
    def __init__(self, Pipe_X, Pipe_Y, Pipe_id,Pipe_Image):
        self.Pipe_X = Pipe_X
        self.Pipe_Lower_Y = Pipe_Y
        self.Pipe_Upper_Y = self.Pipe_Lower_Y - 420
        self.Pipe_id = Pipe_id
        self.Pipe_Lower_Image = Pipe_Image
        self.Pipe_Upper_Image = pygame.transform.flip(self.Pipe_Lower_Image, False, True)

        self.Pipe_Lower_Mask = pygame.mask.from_surface(self.Pipe_Lower_Image)
        self.Pipe_Upper_Mask = pygame.mask.from_surface(self.Pipe_Upper_Image)

    def Draw_Pipe(self,window):
        window.blit(self.Pipe_Lower_Image,(self.Pipe_X, self.Pipe_Lower_Y))
        window.blit(self.Pipe_Upper_Image,(self.Pipe_X, self.Pipe_Upper_Y))


    def Move_Pipe(self, GameSpeed):
        print(self.Pipe_id)
        self.Pipe_X -= 1 * GameSpeed



class GameCore:
    def __init__(self, Population_Number = 1):
        self.window_height = 288
        self.window_width = 512
        self.window = pygame.display.set_mode((self.window_height,self.window_width))
        self.Clock = pygame.time.Clock()
        self.GameSpeed = 2

        self.BackGround = pygame.image.load("assets/background.png").convert()

        self.Base = pygame.image.load("assets/base.png").convert()

        self.Pipe_Image = pygame.image.load("assets/pipe.png").convert_alpha()
        self.Pipe_Number = 0


        self.Bird_Image = pygame.image.load("assets/bird.png").convert_alpha()
        self.Bird_Mask = pygame.mask.from_surface(self.Bird_Image)

        self.Font_T = pygame.font.SysFont("Arial", 40)


        self.Pipe_List = [
            Pipe(300, random.randint(180, 320), 0, self.Pipe_Image),
            Pipe(450, random.randint(180, 320), 1, self.Pipe_Image),
            Pipe(600, random.randint(180, 320), 2, self.Pipe_Image),
            Pipe(750, random.randint(180, 320), 3, self.Pipe_Image)


        ]
        self.Pipe_id = 4


        ### Genetic ####

        self.Population = []
        self.Next_Generation = []
        self.Population_Number = Population_Number
        self.Died_Bird = []
        self.Generation_Timer = 0

        for i in range(self.Population_Number):
            weight_1, weight_2, bias1 , bias2 = self.create_weights()
            self.Population.append(Bird(self.Bird_Image,self.Bird_Mask, weight_1, weight_2, bias1 , bias2))

        self.Deneme = []

    def MaskCollision(self, Masked_Image1, Image1_X, Image1_Y, Masked_Image2, Image2_X, Image2_Y):
        offset = (round(Image2_X - Image1_X), round(Image2_Y - Image1_Y))
        result = Masked_Image1.overlap(Masked_Image2, offset)
        return result

    def Draw(self):

        self.window.blit(self.BackGround,(0,0))
        self.window.blit(self.Base,(0, 400))

        for pipe in self.Pipe_List:
            pipe.Draw_Pipe(self.window)

        for i in self.Population:
            i.Draw_Bird(self.window)

        self.window.blit(self.Font_T.render("" + str(self.Generation_Timer), True, (255, 255, 255)), (20, 20))

        self.Clock.tick(60)
        pygame.display.update()

    def create_weights(self):
        weights1 = np.random.rand(3, 7) - 0.5
        weights2 = np.random.rand(7, 1) - 0.5
        bias1 = np.random.rand(1, 7) - 0.5
        bias2 = np.random.rand(1, 1) - 0.5
        return weights1, weights2, bias1, bias2

    def Crossover(self):
        self.Died_Bird = sorted(self.Died_Bird, key=lambda Bird: Bird.Score)
        if self.Died_Bird[-1].Score == 0:
            self.create_new_generation()
        else:
            self.Next_Generation = []
            last_best = int((95 * self.Population_Number) / 100)
            self.Next_Generation = []
            self.Next_Generation.extend(self.Population[last_best:])
            for Member in self.Next_Generation:
                Member.Bird_X = 50
                Member.Bird_Y = 50
                Member.Score = 0


            while True:
                if len(self.Next_Generation) < self.Population_Number:
                    member_1 = random.choice(self.Died_Bird[last_best:])
                    member_1_weight_1 = member_1.weight_1
                    member_1_weight_2 = member_1.weight_2
                    member_1_bias_1 = member_1.bias_1
                    member_1_bias_2 = member_1.bias_2

                    member_2 = random.choice(self.Died_Bird[last_best:])
                    member_2_weight_1 = member_2.weight_1
                    member_2_weight_2 = member_2.weight_2
                    member_2_bias_1 = member_2.bias_1
                    member_2_bias_2 = member_2.bias_2

                    chield_1_weight_1 = []
                    chield_1_weight_2 = []
                    chield_1_bias_1 = []
                    chield_1_bias_2 = []

                    self.Prob = random.random()
                    for x,y in zip(member_1_weight_1, member_2_weight_1):
                        for i,k in zip(x,y):
                            if self.Prob < 0.47:
                                chield_1_weight_1.append(i)
                            elif self.Prob < 0.94:
                                chield_1_weight_1.append(k)
                            else:
                                chield_1_weight_1.append(random.uniform(-1,1))

                    self.Prob = random.random()
                    for a, b in zip(member_1_weight_2, member_2_weight_2):
                        for c, d in zip(a, b):
                            if self.Prob < 0.47:
                                chield_1_weight_2.append(c)
                            elif self.Prob < 0.94:
                                chield_1_weight_2.append(d)
                            else:
                                chield_1_weight_2.append(random.uniform(-1,1))

                    for  t,y in zip(member_1_bias_1[0], member_2_bias_1[0]):
                        if self.Prob < 0.47:
                            chield_1_bias_1.append(t)
                        elif self.Prob < 0.94:
                            chield_1_bias_1.append(y)
                        else:
                            chield_1_bias_1.append(random.uniform(-1,1))

                    for  q,w in zip(member_1_bias_2, member_2_bias_2):
                        if self.Prob < 0.47:
                            chield_1_bias_2.append(q)
                        elif self.Prob < 0.94:
                            chield_1_bias_2.append(w)
                        else:
                            chield_1_bias_2.append(random.uniform(-1,1))

                    chield_1_weight_1 = np.array(chield_1_weight_1)
                    chield_1_weight_2 = np.array(chield_1_weight_2)

                    chield_1_bias_1 = np.array(chield_1_bias_1)
                    chield_1_bias_2 = np.array(chield_1_bias_2)

                    chield_1_weight_1 = chield_1_weight_1.reshape((3,7))
                    chield_1_weight_2 = chield_1_weight_2.reshape((7,1))
                    chield_1_bias_1 = chield_1_bias_1.reshape((1,7))
                    chield_1_bias_2 = chield_1_bias_2.reshape((1,1))

                    self.Next_Generation.append(Bird(self.Bird_Image,self.Bird_Mask,
                                                     chield_1_weight_1,chield_1_weight_2,
                                                     chield_1_bias_1,chield_1_bias_2))

                else:
                    break
            self.Population = self.Next_Generation

    def create_new_generation(self):
        for i in range(self.Population_Number):
            weight_1, weight_2, bias1, bias2 = self.create_weights()
            self.Population.append(Bird(self.Bird_Image, self.Bird_Mask, weight_1, weight_2, bias1, bias2))

    def restart_game(self):
        self.Pipe_List = [
            Pipe(300, random.randint(180, 320), 0, self.Pipe_Image),
            Pipe(450, random.randint(180, 320), 1, self.Pipe_Image),
            Pipe(600, random.randint(180, 320), 2, self.Pipe_Image),
            Pipe(750, random.randint(180, 320), 3, self.Pipe_Image)

        ]
        self.Pipe_id = 4

        self.Crossover()



    def GameLoop(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "Close"

        self.Tus = pygame.key.get_pressed()
        if self.Tus[pygame.K_ESCAPE]:
            return "Close"

        self.FPS = str(int(self.Clock.get_fps()))
        pygame.display.set_caption(f"Fps : {self.FPS}")

        for pipe in self.Pipe_List:
            if pipe.Pipe_X == -52:
                pipe.Pipe_X = 548
                pipe.Pipe_id = self.Pipe_id
                self.Pipe_id += 1

            pipe.Move_Pipe(self.GameSpeed)



            for Member in self.Population:
                if Member.Score == pipe.Pipe_id:
                    Member.Bird_Distance_With_Pipe = pipe.Pipe_X - Member.Bird_X
                    Member.Pipe_Height = pipe.Pipe_Lower_Y
                    Member.Update_NN()

                collision_lower = self.MaskCollision(Member.Bird_Mask, Member.Bird_X, Member.Bird_Y,
                                                     pipe.Pipe_Lower_Mask,pipe.Pipe_X,pipe.Pipe_Lower_Y)
                collision_upper = self.MaskCollision(Member.Bird_Mask, Member.Bird_X, Member.Bird_Y,
                                                     pipe.Pipe_Upper_Mask,pipe.Pipe_X,pipe.Pipe_Upper_Y)

                if collision_lower != None or collision_upper != None:
                    self.Died_Bird.append(Member)
                    self.Population.remove(Member)

                if Member.Bird_X + 16 == pipe.Pipe_X:
                    Member.Score += 1

        for Member in self.Population:
            if Member.Bird_Loop() == "Died":
                self.Died_Bird.append(Member)
                self.Population.remove(Member)
            Member.Bird_Jump()
        if len(self.Population) == 0:
            self.Generation_Timer += 1
            self.restart_game()

        self.Draw()


Population_Number = 200

Game = GameCore(Population_Number)
while True:
    GameStatus = Game.GameLoop()
    if GameStatus == "Close":
        break

pygame.quit()