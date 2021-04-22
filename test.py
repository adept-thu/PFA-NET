import sys

if __name__ == "__main__":
    name =[]
    i = 0
    for line in sys.stdin:
        a = line.split()
        i = i + 1
        if i == 1:
            num = a
            continue
        name.append(a)
    p = 0
    while len(name)!=0:
        k = name
        num = len(name)
        one = name[0][0]
        two = name[0][1]
        courp = [one, two]
        for j in range(1,num):
            for i in range(1,num):
                if name[i][0] == two:
                    if name[i][1] in courp:
                        break
                    else:
                        one = name[i][0]
                        two = name[i][1]
                        courp.extend([two])
                    k = name

        name_new =[]
        for k in range(num):
            if name[k][0] in courp:
                continue
            else:
                name_new.append([name[k][0],name[k][1]])
        name = name_new
        p = p+1
    print(p)
