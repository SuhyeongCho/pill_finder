from selenium import webdriver
import urllib.request 
from bs4 import BeautifulSoup


driver = webdriver.Chrome('./chromedriver')
driver.implicitly_wait(3)
driver.get('http://www.health.kr/searchIdentity/search.asp')

driver.execute_script('$("#type_01").addClass("selected");')
driver.execute_script('$("#type_01 input:[type=checkbox]").prop("checked",true);')
driver.execute_script('$("#shape_08").addClass("selected");')
driver.execute_script('$("#shape_08 input:[type=checkbox]").prop("checked",true);')


#약학정보원 알약(정제모형) 페이지 수 : 1~337 (16844)
#원형 : 1~ 174 (8673) shape_01
#타원형,장방형 : 1~140 (6999) shape_02,shape_07
#삼각형 1~5 (222) shape_04
#사각형 1~3 (138) shape_05
#마름모형 1~1 (27) shape_06
#오각형 1~2 (65) shape_10
#육각형 1~2 (58) shape_09
#팔각형 1~5 (213) shape_08


for i in range(1,6):
    driver.execute_script('changePage('+str(i)+');')
    print('Change Page to',str(i))
    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')
    #한 페이지에 list 50개씩
    for j in range(1,51):
        try:
            img_tag = soup.select('#idfytotal0 > tbody > tr:nth-child('+str(j+2)+') > td.img > div > a > img')
            url = img_tag[0].get('src')
            savename = './crawling/shape_08/'+str((i-1)*50+j)+'.jpg'
            urllib.request.urlretrieve(url, savename)
            print(savename,' saved')
        except:
            print('finished!')
            break

driver.close()
