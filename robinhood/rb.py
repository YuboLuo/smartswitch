'''
The script reads robinhood transactions from a text file and print out the total gain/loss
You have to first copy all content of the "summary of capital gain" PDF file into a text file
'''

from re import sub
from decimal import Decimal
from collections import defaultdict

with open('rb.txt') as f: # rb.txt is the txt file containing all transactions
    lines = f.readlines()

    total = 0
    table = defaultdict(int)
    for line in lines:
        line = line.split(' ')
        cnt = 0
        for money in line:
            if '$' in money:
                cnt += 1

            if cnt == 4:   # gain or loss is right after the forth '$'

                value = Decimal(sub(r'[^\d\-.]', '', money))
                total += value
                table[line[2]] += int(value)

                print(line[2],money, total) # print out the gain/loss of this transaction and the current total gain/loss
                break

print('\nTotal gain/loss: {}'.format(total))

print('\n(Stock, Gain/Loss)')
for item in sorted(table.items(), key=lambda x: x[1], reverse=True): # sort by the gain of the stock
    print(item)

'''
Output example: 

Total gain/loss: xxx.20

(Stock, Gain/Loss)
('AMC', xxx)
('TESLA,', xxx)
('ALTERYX,', 316)
('JINKOSOLAR', 138)
('ALIBABA', 46)
('TILRAY,', 0)
('UNITY', -2)
('AGENUS', -6)
('GAMESTOP', -158)
('NOKIA', -761)

'''