# Printing some book notes
# Load the db ...

import sqlite3
conn = sqlite3.connect('Copy_AEAnnotation_v10312011_1727_local.sqlite')

print("Book notes:\n")

#SQL query
# Want to get entries where: has highlight, has note
# So we need to select from the table (called 'ZAEANNOTATION', found via DB Browser)
# Sorted by date, right?  recent first, based on https://stackoverflow.com/questions/17411170/sqlite-python-how-to-sort-by-datenewest-first
# have to parse the date - see column in sqlite viewer

cursor = conn.cursor()
# cursor.execute("SELECT * FROM ZAEANNOTATION WHERE ZANNOTATIONNOTE IS NOT NULL ORDER BY ZANNOTATIONLOCATION DESC")
# cursor.execute("SELECT * FROM ZAEANNOTATION WHERE ZANNOTATIONNOTE IS NOT NULL AND ZANNOTATIONSELECTEDTEXT IS NOT NULL ORDER BY ZANNOTATIONMODIFICATIONDATE DESC")
#cursor.execute("SELECT * FROM ZAEANNOTATION WHERE length(ZANNOTATIONNOTE) > length(replace(ZANNOTATIONNOTE, ' ', '')) + 50 AND ZANNOTATIONSELECTEDTEXT IS NOT NULL ORDER BY ZANNOTATIONMODIFICATIONDATE DESC")
# cursor.execute("SELECT * FROM ZAEANNOTATION WHERE length(ZANNOTATIONNOTE) > length(replace(ZANNOTATIONNOTE, ' ', '')) + 25 AND ZANNOTATIONSELECTEDTEXT IS NOT NULL ORDER BY length(ZANNOTATIONNOTE) DESC")
#cursor.execute("SELECT * FROM ZAEANNOTATION WHERE length(ZANNOTATIONNOTE) > length(replace(ZANNOTATIONNOTE, ' ', '')) + 25 AND ZANNOTATIONSELECTEDTEXT IS NOT NULL AND ZANNOTATIONDELETED IS NOT 1 ORDER BY length(ZANNOTATIONNOTE) DESC")
cursor.execute("SELECT * FROM ZAEANNOTATION WHERE length(ZANNOTATIONNOTE) > length(replace(ZANNOTATIONNOTE, ' ', '')) + 15 AND ZANNOTATIONSELECTEDTEXT IS NOT NULL AND ZANNOTATIONDELETED IS NOT 1 ORDER BY ZANNOTATIONMODIFICATIONDATE DESC")


# Still selects highlights with no note content - might be where deleted the content, or put it in as a note but left no text content therein


# Want to print x# most recent entries ... 
# Could replace this with a while loop ... while count is less than 10, do this, incrementing count ... 
count=0
for row in cursor:

	count += 1

	# number of notes to print ... 
	if count>120:
		break

	# could also print row[16] where if it's not there, jsut don't print it ... this was printing the representative text 
	highlighted = row[17] # this is the selected text (whole para) - we would be just looking for the selected highlight, which is +1
	noted = row[15]

	print(count)
	print("Highlighted text:\n" + highlighted + "\n")
	print("Text of note:\n" + noted + "\n")

conn.close()

'''
Now this kind of works ... next steps:
- actually sort by date / recency (parsing the epubcfi - see above selector)
- in the selector, not allowing entries w/ 'none', NULL as ZANNOTATIONNOTE
'''
