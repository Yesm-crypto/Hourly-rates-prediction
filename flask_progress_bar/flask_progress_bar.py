def progress(generator):  
    ''' Generator that takes the progress data into the bar'''
    percent = 0
    while percent < 100:
        progress_data = generator
        for data in progress_data:
            print('Data received - ' + str(data))
            percent = round(((int(data[0])) * 100) / int(data[1]))
            print('Percentage - ' + str(percent))
            yield "data:" + str(percent) + "\n\n"
    yield "data:100\n\n" # in case percentage fails

