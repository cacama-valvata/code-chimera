import datetime
import matplotlib.pyplot as plt
from csAnalyze import *
from csData import *

def plotScansAsDots(data):
	scans = data["scans"]
	startTimestamp = data["startTimestamp"]
	endTimestamp = data["endTimestamp"]
	i=0
	for nodeAddress in scans:
		print len(scans[nodeAddress])
		plt.figure(i)
		timestamps = {}
		for scan in scans[nodeAddress]:
			dev = scan["address"]
			if (dev not in timestamps):
				timestamps[dev] = [scan["time"] - startTimestamp]
			else:
				timestamps[dev].append(scan["time"] - startTimestamp)
		j=0
		for device in timestamps:
			plt.plot(timestamps[device], [j]*len(timestamps[device]), ".")
			j+=1
		names = []
		for device in timestamps:
			names.append(getName(device))
		plt.yticks(range(0,j), names)

		nodeName = getName(nodeAddress)
		plt.title("Devices scanned by " + nodeName)

		duration = endTimestamp-startTimestamp
		xticks = range(0, int(duration+1), int(duration/100))
		formattedTimestamps = []

		for timestamp in xticks:
			formattedTimestamps.append(datetime.datetime.fromtimestamp(timestamp+startTimestamp).strftime("%m-%d %H:%M"))
		plt.xticks(xticks, formattedTimestamps, rotation="vertical")

		plt.grid(axis="x")
#		# See http://stackoverflow.com/questions/12322738/how-do-i-change-the-axis-tick-font-in-a-matplotlib-plot-when-rendering-using-lat#12323891
#		fontDict = {"family":"sans-serif",
#					"sans-serif":["Helvetica"],
#					"weight" : "normal",
#					"size" : 12
#					}
#		fontDict = {
#			"family":"monospace",
#			"size" : 12
#		}
#		gca = plt.gca()
#		gca.set_xticklabels(formattedTimestamps, fontDict)
#		gca.set_yticklabels(names, fontDict)

		plt.axis([0, endTimestamp-startTimestamp, -0.5, j-0.5])
		i+=1


def plotScansAsDots2(data):
	scans = data["scans"]
	startTimestamp = data["startTimestamp"]
	endTimestamp = data["endTimestamp"]
	plt.figure()
	i=0
	names = []
	for nodeAddr in scans:
#		print len(scans[nodeAddr])
#		plt.figure(i)
		timestamps = []
		for scan in scans[nodeAddr]:
			timestamps.append(scan["time"] - startTimestamp)
		plt.plot(timestamps, [i]*len(timestamps), ".")
#		plt.plot(timestamps, [i]*len(timestamps), ",")

		nodeName = getName(nodeAddr)
		names.append(nodeName)
		i+=1

	plt.yticks(range(0,i), names)
	plt.title("Scan nodes")
	plt.axis([0, endTimestamp-startTimestamp, -0.5, i-0.5])



def plotRssiPerDevice(data):
	scans = data["scans"]
	startTimestamp = data["startTimestamp"]
	endTimestamp = data["endTimestamp"]
	data2 = getScansPerDevicePerNode(data)
	scansPerDev = data2["scansPerDev"]
	minRssi = data2["minRssi"]
	maxRssi = data2["maxRssi"]

	figures = []
	lineColors = ["b", "g", "r", "c", "m", "y", "k"]
	lineStyles = ["o", "^", "d", "+", "*"]
	for devAddr in scansPerDev:
#		plt.figure()
		fig, axarr = plt.subplots(len(scansPerDev[devAddr]), sharex=True, sharey=True)
		figures.append(fig)
		devName = getName(devAddr)
#		plt.title("Scanned device: " + devName)
		i=0
		for nodeAddr in scansPerDev[devAddr]:
			timestamps = []
			rssis = []
			for scan in scansPerDev[devAddr][nodeAddr]:
				timestamp = scan["time"]
				rssi = scan["rssi"]
				timestamps.append(timestamp)
				rssis.append(rssi)
#			fmt = lineColors[i % len(lineColors)] + lineStyles[int(i/len(lineColors)) % len(lineStyles)]
			fmt = "b."
			nodeName = getName(nodeAddr)
			subplot = axarr
			if (len(scansPerDev[devAddr]) > 1):
				subplot = axarr[i]
			subplot.plot(timestamps, rssis, fmt, alpha=0.3, label=nodeName)
			subplot.legend(loc="upper left")
#			subplot.set_xlim([startTimestamp, endTimestamp])
			subplot.axis([startTimestamp, endTimestamp, minRssi, maxRssi])
			i+=1
#		plt.legend(loc="upper left")

		duration = endTimestamp-startTimestamp
		xticks = range(int(startTimestamp), int(endTimestamp+1), int(duration/100))
		formattedTimestamps = []
		for xtick in xticks:
			formattedTimestamps.append(datetime.datetime.fromtimestamp(xtick).strftime("%m-%d %H:%M"))

		if (len(scansPerDev[devAddr]) > 1):
			fig.subplots_adjust(hspace=0)
			axarr[0].set_title("Scanned device: " + devName)
			axarr[-1].set_xticks(xticks)
			axarr[-1].set_xticklabels(formattedTimestamps, rotation="vertical")
			plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
		else:
			axarr.set_title("Scanned device: " + devName)
			axarr.set_xticks(xticks)
			axarr.set_xticklabels(formattedTimestamps, rotation="vertical")
	return figures



def plotRssiPerNode(data):
	scans = data["scans"]
	startTimestamp = data["startTimestamp"]
	endTimestamp = data["endTimestamp"]

	data2 = getScansPerNodePerDevice(data)
	scansPerNode = data2["scansPerNode"]
	minRssi = data2["minRssi"]
	maxRssi = data2["maxRssi"]

	figures = []
	lineColors = ["b", "g", "r", "c", "m", "y", "k"]
	lineStyles = ["o", "^", "d", "+", "*"]
	for nodeAddr in scansPerNode:
#		plt.figure()
		fig, axarr = plt.subplots(len(scansPerNode[nodeAddr]), sharex=True, sharey=True)
		figures.append(fig)
		nodeName = getName(nodeAddr)
#		plt.title("Node: " + nName)
		i=0
		for devAddr in scansPerNode[nodeAddr]:
			timestamps = []
			rssis = []
			for scan in scansPerNode[nodeAddr][devAddr]:
				timestamp = scan["time"]
				rssi = scan["rssi"]
				timestamps.append(timestamp)
				rssis.append(rssi)
#			fmt = lineColors[i % len(lineColors)] + lineStyles[int(i/len(lineColors)) % len(lineStyles)]
			fmt = "b."
			devName = getName(devAddr)
			subplot = axarr
			if (len(scansPerNode[nodeAddr]) > 1):
				subplot = axarr[i]
			subplot.plot(timestamps, rssis, fmt, alpha=0.3, label=devName)
			subplot.legend(loc="upper left")
#			subplot.set_xlim([startTimestamp, endTimestamp])
			subplot.axis([startTimestamp, endTimestamp, minRssi, maxRssi])
			i+=1
#		plt.legend(loc="upper left")

		duration = endTimestamp-startTimestamp
		xticks = range(int(startTimestamp), int(endTimestamp+1), int(duration/100))
		formattedTimestamps = []
		for xtick in xticks:
			formattedTimestamps.append(datetime.datetime.fromtimestamp(xtick).strftime("%m-%d %H:%M"))

		if (len(scansPerNode[nodeAddr]) > 1):
			fig.subplots_adjust(hspace=0)
			axarr[0].set_title("Scanned device: " + nodeName)
			axarr[-1].set_xticks(xticks)
			axarr[-1].set_xticklabels(formattedTimestamps, rotation="vertical")
			plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
		else:
			axarr.set_title("Scanned device: " + nodeName)
			axarr.set_xticks(xticks)
			axarr.set_xticklabels(formattedTimestamps, rotation="vertical")
	return figures



def plotScanFrequency(data, windowSize=600, stepSize=60):
	scans = data["scans"]
	startTimestamp = data["startTimestamp"]
	endTimestamp = data["endTimestamp"]

	data2 = getFrequencyPerDevicePerNode(data, windowSize, stepSize)
	numScans = data2["numScansPerDev"]
	startTimes = data2["startTimes"]

	lineColors = ["b", "g", "r", "c", "m", "y", "k"]
	lineStyles = ["o", "^", "d", "+", "*"]

	figures = []
	for devAddr in numScans:
#		plt.figure()
		fig, axarr = plt.subplots(len(numScans[devAddr]), sharex=True, sharey=True)
		figures.append(fig)
		devName = getName(devAddr)

		i=0
		for nodeAddr in numScans[devAddr]:
#			fmt = lineColors[i % len(lineColors)] + lineStyles[int(i/len(lineColors)) % len(lineStyles)]
			fmt = "b-"
			nodeName = getName(nodeAddr)
			subplot = axarr[i]
			if (len(numScans[devAddr]) < 2):
				subplot = axarr
			subplot.plot(startTimes[1:], numScans[devAddr][nodeAddr], fmt, alpha=1.0, label=nodeName)
			subplot.legend(loc="upper left")
			subplot.set_xlim([startTimestamp, endTimestamp])
			i+=1

		duration = endTimestamp-startTimestamp
		xticks = range(int(startTimestamp), int(endTimestamp+1), int(duration/100))
		formattedTimestamps = []
		for xtick in xticks:
			formattedTimestamps.append(datetime.datetime.fromtimestamp(xtick).strftime("%m-%d %H:%M"))
		if (len(numScans[devAddr]) > 1):
			fig.subplots_adjust(hspace=0)
			axarr[0].set_title("Scanned device: " + devName)
			axarr[-1].set_xticks(xticks)
			axarr[-1].set_xticklabels(formattedTimestamps, rotation="vertical")
			plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
		else:
			axarr.set_title("Scanned device: " + devName)
			axarr.set_xticks(xticks)
			axarr.set_xticklabels(formattedTimestamps, rotation="vertical")
	return figures



def plotAvgRssi(data, windowSize=600, stepSize=60):
	scans = data["scans"]
	startTimestamp = data["startTimestamp"]
	endTimestamp = data["endTimestamp"]

	data2 = getAverageRssiPerDevicePerNode(data, windowSize, stepSize)
	avgRssi = data2["avgRssiPerDev"]
	startTimes = data2["startTimes"]

	lineColors = ["b", "g", "r", "c", "m", "y", "k"]
	lineStyles = ["o", "^", "d", "+", "*"]

	figures = []
	for devAddr in avgRssi:
#		plt.figure()
		fig, axarr = plt.subplots(len(avgRssi[devAddr]), sharex=True, sharey=True)
		figures.append(fig)
		devName = getName(devAddr)

		i=0
		for nodeAddr in avgRssi[devAddr]:
#			fmt = lineColors[i % len(lineColors)] + lineStyles[int(i/len(lineColors)) % len(lineStyles)]
			fmt = "b-"
			nodeName = getName(nodeAddr)
#			plt.plot(startTimes[1:], numScans[nodeAddr], fmt, alpha=0.3, label=nodeName)
			subplot = axarr[i]
			if (len(avgRssi[devAddr]) < 2):
				subplot = axarr
#			subplot.plot(startTimes[1:], avgRssi[devAddr][nodeAddr], fmt, alpha=1.0, label=nodeName)
			subplot.plot(startTimes, avgRssi[devAddr][nodeAddr], fmt, alpha=1.0, label=nodeName)
			subplot.legend(loc="upper left")
			subplot.set_xlim([startTimestamp, endTimestamp])
			i+=1

		duration = endTimestamp-startTimestamp
		xticks = range(int(startTimestamp), int(endTimestamp+1), int(duration/100))
		formattedTimestamps = []
		for xtick in xticks:
			formattedTimestamps.append(datetime.datetime.fromtimestamp(xtick).strftime("%m-%d %H:%M"))
		if (len(avgRssi[devAddr]) > 1):
			fig.subplots_adjust(hspace=0)
			axarr[0].set_title("Scanned device: " + devName)
			axarr[-1].set_xticks(xticks)
			axarr[-1].set_xticklabels(formattedTimestamps, rotation="vertical")
			plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
		else:
			axarr.set_title("Scanned device: " + devName)
			axarr.set_xticks(xticks)
			axarr.set_xticklabels(formattedTimestamps, rotation="vertical")
	return figures



def plotBandwidth(data):
	# TODO: last value of numScans is wrong because that timeslot can be less than dt

	dt = 60

	scans = data["scans"]
	startTimestamp = data["startTimestamp"]
	endTimestamp = data["endTimestamp"]
	startTimes = range(int(startTimestamp), int(endTimestamp)+dt, dt)
#	print startTimes
	numScans = {}
	numScans["total"] = [0]*len(startTimes)

	lineColors = ["b", "g", "r", "c", "m", "y", "k"]
	lineStyles = ["-o", "-^", "-d", "-+", "-*"]

	plt.figure()
	i = 0
	for node in scans:
		numScans[node] = [0]*len(startTimes)
		timeInd = 0
		for scan in scans[node]:
			scanTime = int(scan["time"])
			if (scanTime > startTimes[timeInd+1]):
				timeInd += 1
#			numScans[timeInd] += 1
			numScans[node][timeInd] += 1.0 / dt
			numScans["total"][timeInd] += 1.0 / dt
			scannerName = getName(node)
		plt.title("Number of scanned devices per second")
		fmt = lineColors[i % len(lineColors)] + lineStyles[int(i/len(lineColors)) % len(lineStyles)]
		plt.plot(startTimes[0:-1], numScans[node][0:-1], fmt, label=scannerName)
		i+=1
	plt.legend(loc="upper left")

	plt.figure()
	plt.title("Total number of scanned devices per second")
	plt.plot(startTimes[0:-1], numScans["total"][0:-1])

def plotBandwidth2(data):
	# Time step
	dt = 60

	# Averaging window: must be divisible by 2*dt
	window = 300

	scans = data["scans"]
	startTimestamp = data["startTimestamp"]
	endTimestamp = data["endTimestamp"]
	startTimes = range(int(startTimestamp), int(endTimestamp)+dt, dt)
#	print startTimes
	numScans = {}
	numScans["total"] = [0]*len(startTimes)

	plt.figure()
	for node in scans:
		numScans[node] = [0]*len(startTimes)
		timeInd = 0
		for scan in scans[node]:
			scanTime = int(scan["time"])

			if (scanTime > startTimes[timeInd+1]):
				timeInd += 1

			tIndStart = max(timeInd - window/2/dt, 0)
			tIndEnd = min(timeInd + window/2/dt, len(startTimes)-1)
			for tInd in xrange(tIndStart, tIndEnd):
				numScans[node][tInd] += 1.0
				numScans["total"][tInd] += 1.0

		for timeInd in xrange(0, len(startTimes)):
			tIndStart = max(timeInd - window/2/dt, 0)
			tIndEnd = min(timeInd + window/2/dt, len(startTimes)-1)
			actualWindowSize = startTimes[tIndEnd] - startTimes[tIndStart]
			numScans[node][timeInd] /= actualWindowSize

		plt.title("Number of scanned devices per second (moving average)")
		plt.plot(startTimes[0:-1], numScans[node][0:-1])

	for timeInd in xrange(0, len(startTimes)):
		tIndStart = max(timeInd - window/2/dt, 0)
		tIndEnd = min(timeInd + window/2/dt, len(startTimes)-1)
		actualWindowSize = startTimes[tIndEnd] - startTimes[tIndStart]
		numScans["total"][timeInd] /= actualWindowSize

	plt.figure()
	plt.title("Total number of scanned devices per second (moving average)")
	plt.plot(startTimes[0:-1], numScans["total"][0:-1])


if __name__ == '__main__':
	print "File not intended as main."