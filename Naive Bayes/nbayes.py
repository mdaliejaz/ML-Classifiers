from collections import defaultdict
from math import log


def naive_bayes():
    flabel = open("traininglabels.txt", 'r')
    fimage = open("trainingimages.txt", 'r')

    training_data_count = 0
    training_dict = defaultdict(list)
    image_digit = []
    for digit_line in flabel:
        digit = int(digit_line)
        training_data_count += 1
        for i in range(28):
            image_digit_line = list(fimage.readline().rstrip('\n'))
            image_digit.append(image_digit_line)
        training_dict[digit].append(image_digit)
        image_digit = []

    training_probability = defaultdict(lambda: defaultdict(list))
    probability_digit = defaultdict(list)
    for digit in range(10):
        probability_white = [[0 for x in range(28)] for x in range(28)]
        probability_dark = [[0 for x in range(28)] for x in range(28)]
        digit_images = training_dict[digit]
        image_count = len(digit_images)
        probability_digit[digit].append(image_count / (training_data_count * 1.0))
        for image in digit_images:
            for i in range(28):
                for j in range(28):
                    if image[i][j] == '#' or image[i][j] == '+':
                        probability_dark[i][j] += 6
                    else:
                        probability_white[i][j] += 1
        for i in range(28):
            for j in range(28):
                probability_white[i][j] /= (image_count * 1.0)
                probability_dark[i][j] /= (image_count * 1.0)
        training_probability[digit]['w'].append(probability_white)
        training_probability[digit]['d'].append(probability_dark)
    fimage.close()
    flabel.close()
    tlabel = open('testlabels.txt', 'r')
    timage = open('testimages.txt', 'r')
    predict_label = open("output.txt", 'w')

    smoothing_value = 1
    test_data_count = 0
    log_probability = 0
    max_so_far = -1000
    test_image_digit = []
    digit_predicted = -1
    for test_digit_line in tlabel:
        test_data_count += 1
        for i in range(28):
            test_image_digit_line = list(timage.readline().rstrip('\n'))
            test_image_digit.append(test_image_digit_line)
        for num in range(10):
            for i in range(28):
                for j in range(28):
                    if test_image_digit[i][j] == ' ':
                        log_probability += log(training_probability[num]['w'][0][i][j] + smoothing_value)
                    else:
                        log_probability += log(training_probability[num]['d'][0][i][j] + smoothing_value)
                    if j + 1 < 28:
                        if test_image_digit[i][j] == ' ' and test_image_digit[i][j + 1] == '#':
                            log_probability += log(training_probability[num]['d'][0][i][j] + smoothing_value)
                    elif j - 1 >= 0:
                        if test_image_digit[i][j] == ' ' and test_image_digit[i][j - 1] == '#':
                            log_probability += log(training_probability[num]['d'][0][i][j] + smoothing_value)
                    if j + 1 < 28:
                        if test_image_digit[i][j] == ' ' and test_image_digit[i][j + 1] == '+':
                            log_probability += log(training_probability[num]['d'][0][i][j] + smoothing_value)
                    elif j - 1 >= 0:
                        if test_image_digit[i][j] == ' ' and test_image_digit[i][j - 1] == '+':
                            log_probability += log(training_probability[num]['d'][0][i][j] + smoothing_value)
            log_probability += log(probability_digit[num][0] + smoothing_value)
            if max_so_far < log_probability:
                digit_predicted = num
                max_so_far = log_probability
            log_probability = 0
        predict_label.write("%d\n" % digit_predicted)
        max_so_far = -1
        test_image_digit = []
    timage.close()
    tlabel.close()
    predict_label.close()

    test_label = open('testlabels.txt', 'r')
    predict_label = open("output.txt", 'r')
    correct_sum = 0
    total_sum = 0
    total_num_digits = []
    correct_match_digits = []
    buckets = []

    for i in xrange(10):
        total_num_digits.append(0)
        correct_match_digits.append(0)
    for test_digit_line in test_label:
        actual_digit = int(test_digit_line)
        total_num_digits[actual_digit] += 1
        predicted_digit = int(predict_label.readline().rstrip('\n'))
        if actual_digit == predicted_digit:
            correct_match_digits[actual_digit] += 1
    print "Number of each digit in test data and its corresponding match number:"
    for i in xrange(10):
        print "Digit", i, ":", "Test data occurrence:", total_num_digits[i], "; Matched successfully:", correct_match_digits[i]
        total_sum += total_num_digits[i]
        correct_sum += correct_match_digits[i]
    print

    print "Percentage precision for each digit:"
    for i in xrange(10):
        print "Digit", i, ":",
        print (correct_match_digits[i]*100) / float(total_num_digits[i]), "%"
    print

    print "Total number of digits in test data:", total_sum
    print "Total number of matched digits:", correct_sum
    print "Overall percentage of precision:", (correct_sum * 100) / float(total_sum), "%"

    test_label.close()
    predict_label.close()


if __name__ == '__main__':
    naive_bayes()