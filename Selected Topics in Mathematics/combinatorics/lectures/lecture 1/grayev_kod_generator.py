
# Generator Grayevog koda s n clanova.

def gray_kod(n):
    lijevi = ['0', '1']
    while(len(lijevi) < n):
        desni = lijevi[:]
        desni.reverse()

        desni = ['1' + rijec for rijec in desni]
        lijevi = ['0' + rijec for rijec in lijevi]

        lijevi += desni

    return lijevi[:n + 1]


def main():
    print(gray_kod(100))


main()
