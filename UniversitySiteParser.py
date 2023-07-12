import requests
from lxml import etree
from io import StringIO

urls = ['https://priemsamara.ru/ratings/?pk=1003&pay=budget&filter=all',
        'https://priemsamara.ru/ratings/?pk=1012&pay=budget&filter=all']


class StudentProfile:
    def __init__(self, id: str, testResult: int, diplomaAvailability: str, priority: int):
        self.id = id
        self.testResult = testResult
        self.diploma = diplomaAvailability == 'Да'
        self.priority = priority


class UniversityData:
    def __init__(self, url: str):
        self.url = url
        self.name = None
        self.studTable: list[StudentProfile] = list()

    def get_full_data(self):
        data = get_parsable_site(self.url)

        name = data.xpath('/html/body/div[1]/div[2]/div[1]/div/div[1]/label[2]')
        self.name = ' '.join(name[0].text.split())

        priorityXPath = '//*[@id="mag_table_id"]/tr/td[6]'
        diplomaAvailabilityXPath = '//*[@id="mag_table_id"]/tr/td[5]'
        testResultXPath = '//*[@id="mag_table_id"]/tr/td[4]'
        idXPath = '//*[@id="mag_table_id"]/tr/td[2]'

        priorityData = data.xpath(priorityXPath)
        diplomaAvailabilityData = data.xpath(diplomaAvailabilityXPath)
        testResultData = data.xpath(testResultXPath)
        idData = data.xpath(idXPath)

        for i in range(len(idData)):
            self.studTable.append(StudentProfile(idData[i].text,
                                                 int(testResultData[i].text),
                                                 diplomaAvailabilityData[i].text.strip(),
                                                 int(priorityData[i].text)))

    def __str__(self):
        return f'Направление: {self.name}\nКоличество абитуриентов: {len(self.studTable)}\n' \
               f'Приоритеты: {self.get_priority_amount(self.studTable)}\nПриоритеты с дипломами: ' \
               f'{self.get_priority_amount_with_diploma()}\n' \
               f'Количество людей с оригиналами дипломов: ' \
               f'{self.get_diploma_amount()}\n'

    def get_priority_amount(self, table):
        priorities = {}
        for stud in table:
            if stud.priority in priorities.keys():
                priorities[stud.priority] += 1
            else:
                priorities[stud.priority] = 1
        return priorities

    def get_diploma_amount(self):
        return sum([1 for i in self.studTable if i.diploma])

    def get_priority_amount_with_diploma(self):
        relevantData = [i for i in self.studTable if i.diploma]
        return self.get_priority_amount(relevantData)


def get_parsable_site(url: str):
    request = requests.get(url)
    return etree.parse(StringIO(request.content.decode('utf-8')), etree.HTMLParser())


if __name__ in '__main__':
    universitiesData = [UniversityData(f'https://priemsamara.ru/ratings/?pk={i}&pay=budget&filter=all')
                        for i in range(1002, 1020)]
    for uni in universitiesData:
        uni.get_full_data()
        print(uni)
