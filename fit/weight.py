import matplotlib.pyplot as plt
import datetime as dt
import xlrd

wb = xlrd.open_workbook("C:\\Users\\Mog\\Desktop\\fit.xlsx")
sheet = wb.sheet_by_index(0)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_title('Fitness Journey')
ax1.set_xlabel('Dias')
ax1.set_ylabel('Peso')
ax2.set_ylabel('Perda de peso')

ax1.set_xlim(left = 0)
ax1.set_xlim(right = 10)

ax1.set_ylim(top=80)
ax1.set_ylim(bottom=75)
ax2.set_ylim(top=1)
ax2.set_ylim(bottom=0)


y = []   # Peso Diario começa dia 2
y2 = []  #diferença de peso começa dia 1
x = []
x2 = []
for i in range(2,sheet.nrows):
    y.append(float(sheet.cell_value(i,1)))

for i in range(1,sheet.nrows):
    y2.append(float(sheet.cell_value(i,4)))

for i in range(2,sheet.nrows):
    x.append(float(sheet.cell_value(i,0)))    

for i in range(1,sheet.nrows):
    x2.append(float(sheet.cell_value(i,0))) 
 

ax1.plot(x,y, marker='o', color='b', label='Peso Diario(Kg)')
ax2.scatter(x2,y2, marker='o', color='r', label='Perda por treino(Kg)')
ax1.yaxis.label.set_color('blue')
ax2.yaxis.label.set_color('red')
ax1.legend(loc=2)
ax2.legend(loc=1)

plt.show()
