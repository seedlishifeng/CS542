'''
Shifeng Li
U24567277

Sicssor   1
Rock      2
Paper     3

'''
import os
import numpy
import random

file = open('DATA','r')
DATA = [line.split(' ')for line in file]
DATA = [[line2.replace('\n','') for line2 in line] for line in DATA]
transform_data = numpy.zeros([len(DATA),2])
for i in range(len(DATA)):
    if DATA[i][1] =='P':
        transform_data[i][1]=3
    if DATA[i][1] =='R':
        transform_data[i][1]=2
    if DATA[i][1] =='S':
        transform_data[i][1]=1
    if DATA[i][0] =='P':
        transform_data[i][0]=3
    if DATA[i][0] =='R':
        transform_data[i][0]=2
    if DATA[i][0] =='S':
        transform_data[i][0]=1
#print(transform_data)

def history(M):
    N=len(M)
    person_win = 0
    computer_win = 0
    both_tie = 0
    person1=[]
    person2=[]
    person3=[]
    compuer1=[]
    compuer2 = []
    compuer3 = []
    for i in range(N):
        result = (int(M[i][0])-int(M[i][1])+4) % 3 -1
        if result > 0:
            person_win = person_win + 1
            if M[i][0]==1:
                person1.append(M[i][0])
            if M[i][0] == 2:
                person2.append(M[i][0])
            if M[i][0] == 1:
                person3.append(M[i][0])
        elif result == 0:
            both_tie= both_tie + 1
        elif result < 0:
            computer_win = computer_win + 1
            if M[i][0]==1:
                compuer1.append(M[i][1])
            if M[i][0] == 2:
                compuer2.append(M[i][1])
            if M[i][0] == 3:
                compuer3.append(M[i][1])
    return person1,person2,person3,compuer1,compuer2,compuer3,person_win,computer_win,both_tie
'''''
a,b,c,d,e,f,g,h,i=history(transform_data)
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(i)
'''
def probability(M,counta,countb,countc,countd,counte,countf,countg):
    A, B, C, D, E, F, G, H, I=history(M)
    #if the probability of person win,then the probability of person choose S to win
    winS_person = len(A)+counta
    win_person = G+counta+countb+countc
    winR_person = len(B)+countb
    winP_person = len(C)+countc
    win_computer = H + countd + counte + countf
    winS_computer = len(D)+countd
    winR_computer = len(E)+counte
    winP_computer = len(F)+countf
    tie = countg + I
    return winS_person,winR_person,winP_person,winS_computer,winR_computer,winP_computer,win_person,win_computer,tie

def Play(M):
    user = int(input())
    you_win_count = 0
    you_wins= 0
    you_winr= 0
    you_winp =0
    tie_count = 0
    computer_win_count = 0
    computer_wins= 0
    computer_winr= 0
    computer_winp= 0
    select = ['S', 'R', 'P']
    count = 0
    A, B, C, D, E, F, G, H, I = probability(M,you_wins,you_winr,you_winp,computer_wins,computer_winr,computer_winp,tie_count)
    computer = random.randint(1, 3)
    while user != 0:
        if 3 < user or user < 1:
            print ('It is wrong ')
        else:
            count = count + 1
            print('you are' + str(select[user - 1]))
            print('computer is' + str(select[computer - 1]))
            result = (user - computer + 4) % 3 - 1
        if result > 0:
            you_win_count = you_win_count + 1
            if computer==1:
                A=A+1
                tmp = random.randint(1, A + B + C)
                if tmp>1 and tmp<=A:
                    choose = 2
                if tmp>A and tmp<= A+B:
                    choose = 3
                if tmp > A+B and tmp <= A+B+C:
                    choose = 1
            if computer ==2:
                B = B+1
                tmp = random.randint(1, A + B + C)
                if tmp>1 and tmp<=A:
                    choose = 2
                if tmp>A and tmp<= A+B:
                    choose = 3
                if tmp > A+B and tmp <= A+B+C:
                    choose = 1

            if computer==3:
                C = C+1
                tmp = random.randint(1, A + B + C)
                if tmp > 1 and tmp <= A:
                    choose = 2
                if tmp > A and tmp <= A + B:
                    choose = 3
                if tmp > A + B and tmp <= A + B + C:
                    choose = 1
            computer = choose
            print('You Win')
        elif result == 0:
             tie_count = tie_count +1
             print('Tie')
        elif result < 0:
             computer_win_count=computer_win_count+1
             if computer == 1:
                 A = A + 1
                 tmp = random.randint(1, F+D+E)
                 if tmp > 1 and tmp <= D:
                     choose = 1
                 if tmp > D and tmp <= E + D:
                     choose = 2
                 if tmp > E + D and tmp <= F+D+E:
                     choose = 3
             if computer == 2:
                 B = B + 1
                 tmp = random.randint(1, F+D+E)
                 if tmp > 1 and tmp <= D:
                     choose = 1
                 if tmp > D and tmp <= D+E:
                     choose = 2
                 if tmp > D+E and tmp <= D+E+F:
                     choose = 3

             if computer == 3:
                 C = C + 1
                 tmp = random.randint(1, F+D+E)
                 if tmp > 1 and tmp <= D:
                     choose = 1
                 if tmp > D and tmp <= D+E:
                     choose = 2
                 if tmp > D+E and tmp <= D+E+F:
                     choose = 3
             computer = choose
             print('Computer Win')
        go_on = input("Continue?[Y/N]:")
        if go_on == 'Y' or go_on == 'y':
             user = int(input())
        else:
            print(G)
            print(H)
            print(I)
            break
Play(transform_data)
