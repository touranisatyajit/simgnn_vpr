import csv
import random
with open('train.csv', 'w') as file:
    employee_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(10000):
        print(i)
        if(i % 2 == 1):
            x = random.randint(20, 12800)
            y = x + random.randint(-10, 10)
            employee_writer.writerow([str(x), str(y), str(1),'\n'])
        else:
            x = random.randint(20, 12800)
            y = -1
            if(i % 4 == 0):
                y = x + random.randint(-12800,-20)
                while(True):
                    if(y >= 2):
                        break
                    y = x + random.randint(-12800,-20)
                employee_writer.writerow([str(x), str(y), str(0), '\n'])
            else:
                y = x + random.randint(20, 12800)
                while(True):
                    if(y <= 12800):
                        break
                    y = x + random.randint(20, 12800)
                employee_writer.writerow([str(x), str(y),  str(0),'\n']) 

