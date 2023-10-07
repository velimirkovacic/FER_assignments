def namjesti_znamenku(broj, index_znamenke, index1, index2):

    if(broj[index_znamenke == '9' and index_znamenke == 0]):
        return None

    if(broj[index_znamenke] == '9' and index_znamenke != 0):
        broj[index_znamenke] = '0'
        return namjesti_znamenku(broj, index_znamenke - 1, index1, index2)

    if broj[index_znamenke] == '0' and index1 == -1 and (index2 == -1 or index_znamenke <= index2):
        broj[index_znamenke] = '1'
        return broj

    if broj[index_znamenke] == '0' and index2 == -1:
        broj[index_znamenke] = '2'
        return broj

    if broj[index_znamenke] == '1' and index2 == -1 and (index1 == -1 or index_znamenke >= index1):
        broj[index_znamenke] = '2'
        return broj
    
    if broj[index_znamenke] in {'0', '1'}:
        broj[index_znamenke] = '3'
        return broj

    broj[index_znamenke] = str(int(broj[index_znamenke]) + 1)
    return broj


def sljedbenik(broj):
    broj = str(broj)
    return int("".join(namjesti_znamenku(list(broj), len(broj) - 1, broj.find("1"), broj.find("2"))))


def main():
    broj = 10000
    brojac = 0

    while(broj != None):
        brojac += 1
        broj = sljedbenik(broj)

    print("Takvih brojeva ima: ", brojac)


main()