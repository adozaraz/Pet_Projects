import itertools

def parseData():
    size = int(input())
    friends = [set([int(i) for i in input().split()]) for _ in range(size)]
    return size, friends


def main():
    size, friends = parseData()
    result = 1
    for i in friends:
        result += len(i)
    print(result)



if __name__ in '__main__':
    main()
