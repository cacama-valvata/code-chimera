# -*- coding: utf-8 -*-

import time
import platform
import os
import sys
import numpy as np
import math
import concurrent.futures
is_frozen = getattr(sys, 'frozen', False)

def subdivide(length, parts=None, max_length=None):
	"""Generates a list with start end stop indices of length parts, [(0, length/parts), ..., (.., length)]"""
	if max_length:
		i1 = 0
		done = False
		while not done:
			i2 = min(length, i1 + max_length)
			#print i1, i2
			yield i1, i2
			i1 = i2
			if i1 == length:
				done = True
	else:
		part_length = int(math.ceil(float(length)/parts))
		#subblock_count = math.ceil(total_length/subblock_size)
		#args_list = []
		for index in range(parts):
			i1, i2 = index * part_length, min(length, (index +1) * part_length)
			yield i1, i2


def submit_subdivide(thread_count, f, length, max_length):
	futures = []
	thread_pool = concurrent.futures.ThreadPoolExecutor(thread_count)
	#thread_pool = concurrent.futures.ProcessPoolExecutor(thread_count)
	for i1, i2 in list(subdivide(length, max_length=max_length)):
		futures.append(thread_pool.submit(f, i1, i2))
	return futures

def linspace_centers(start, stop, N):
	return np.arange(N) / (N+0.) * (stop-start) + float(stop-start)/N/2 + start

def multisum(a, axes):
	correction = 0
	for axis in axes:
		a = np.nansum(a, axis=axis-correction)
		correction += 1
	return a

def disjoined(data):
	# create marginalized distributions and multiple them together
	data_disjoined = None
	dim = len(data.shape)
	for d in range(dim):
		axes = list(range(dim))
		axes.remove(d)
		data1d = multisum(data, axes)
		shape = [1 for k in range(dim)]
		shape[d] = len(data1d)
		data1d = data1d.reshape(tuple(shape))
		if d == 0:
			data_disjoined = data1d
		else:
			data_disjoined = data_disjoined * data1d
	return data_disjoined



def get_data_file(filename):
	try: # this works for egg like stuff, but fails for py2app apps
		from pkg_resources import Requirement, resource_filename
		path = resource_filename(Requirement.parse("vaex"), filename)
		if os.path.exists(path):
			return path
	except:
		pass
	# this is where we expect data to be in normal installations
	for extra in ["", "data", "data/dist", "../data", "../data/dist"]:
		path = os.path.join(os.path.dirname(__file__), "..", "..", extra, filename)
		if os.path.exists(path):
			return path
		path = os.path.join(sys.prefix, extra, filename)
		if os.path.exists(path):
			return path
		# if all fails..
		path = os.path.join(get_root_path(), extra, filename)
		if os.path.exists(path):
			return path

def get_root_path():
	osname = platform.system().lower()
	#if (osname == "linux") and is_frozen: # we are using pyinstaller
	if is_frozen: # we are using pyinstaller or py2app
		return os.path.dirname(sys.argv[0])
	else:
		return os.path.abspath(".")
	
	
def os_open(document):
	"""Open document by the default handler of the OS, could be a url opened by a browser, a text file by an editor etc"""
	osname = platform.system().lower()
	if osname == "darwin":
		os.system("open \"" +document +"\"")
	if osname == "linux":
		cmd = "xdg-open \"" +document +"\"&"
		os.system(cmd)
	if osname == "windows":
		os.system("start \"" +document +"\"")

def filesize_format(value):
	for unit in ['bytes','KiB','MiB','GiB']:
		if value < 1024.0:
			return "%3.1f%s" % (value, unit)
		value /= 1024.0
	return "%3.1f%s" % (value, 'TiB')



log_timer = True
class Timer(object):
	def __init__(self, name=None, logger=None):
		self.name = name
		self.logger = logger

	def __enter__(self):
		if log_timer:
			if self.logger:
				self.logger.debug("%s starting" % self.name)
			else:
				print(('[%s starting]...' % self.name))
			self.tstart = time.time()

	def __exit__(self, type, value, traceback):
		if log_timer:
			msg = "%s done, %ss elapsed" % (self.name, time.time() - self.tstart)
			if self.logger:
				self.logger.debug(msg)
			else:
				print(msg)
			if type or value or traceback:
				print((type, value, traceback))
		return False

def get_private_dir():
	path = os.path.expanduser('~/.vaex')
	if not os.path.exists(path):
		os.makedirs(path)
	return path

def make_list(sequence):
	if isinstance(sequence, np.ndarray):
		return sequence.tolist()
	else:
		return list(sequence)

import progressbar
#from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
#   FileTransferSpeed, FormatLabel, Percentage, \
#    ProgressBar, ReverseBar, RotatingMarker, \
#    SimpleProgress, Timer, AdaptiveETA, AbsoluteETA, AdaptiveTransferSpeed
#from progressbar.widgets import TimeSensitiveWidgetBase, FormatWidgetMixin

class CpuUsage(progressbar.widgets.FormatWidgetMixin, progressbar.widgets.TimeSensitiveWidgetBase):
	def __init__(self, format='CPU Usage: %(cpu_usage)s%%', usage_format="% 5d"):
		super(CpuUsage, self).__init__(format=format)
		self.usage_format = usage_format
		self.utime_0 = None
		self.stime_0 = None
		self.walltime_0 = None

	def __call__(self, progress, data):
		utime, stime, child_utime, child_stime, walltime = os.times()
		if self.utime_0 is None:
			self.utime_0 = utime
		if self.stime_0 is None:
			self.stime_0 = stime
		if self.walltime_0 is None:
			self.walltime_0 = walltime
		data["utime_0"] = self.utime_0
		data["stime_0"] = self.stime_0
		data["walltime_0"] = self.walltime_0

		delta_time = utime - self.utime_0 + stime - self.stime_0
		delta_walltime = walltime - self.walltime_0
		#print delta_time, delta_walltime, utime, self.utime_0, stime, self.stime_0
		if delta_walltime == 0:
			data["cpu_usage"] = "---"
		else:
			cpu_usage = delta_time/(delta_walltime * 1.) * 100
			data["cpu_usage"] = self.usage_format % cpu_usage
		#utime0, stime0, child_utime0, child_stime0, walltime0 = os.times()
		return progressbar_mod.widgets.FormatWidgetMixin.__call__(self, progress, data)
progressbar_mod = progressbar
def progressbar(name="processing", max_value=1):
	widgets = [
		name,
        ': ', progressbar_mod.widgets.Percentage(),
        ' ', progressbar_mod.widgets.Bar(),
        ' ', progressbar_mod.widgets.ETA(),
        #' ', progressbar_mod.widgets.AdaptiveETA(),
		' ', CpuUsage()
    ]
	bar = progressbar_mod.ProgressBar(widgets=widgets, max_value=max_value)
	bar.start()
	return bar
	#FormatLabel('Processed: %(value)d lines (in: %(elapsed)s)')

def progressbar_callable(name="processing", max_value=1):
	bar = progressbar(name, max_value=max_value)
	def update(fraction):
		bar.update(fraction)
		if fraction == 1:
			bar.finish()
		return True
	return update


def confirm_on_console(topic, msg):
	done = False
	print(topic)
	while not done:
		output = raw_input(msg +":[y/n]")
		if output.lower() == "y":
			return True
		if output.lower() == "n":
			return False

import json
import yaml
def write_json_or_yaml(filename, data):
	base, ext = os.path.splitext(filename)
	if ext == ".json":
		json.dump(data, filename)
	elif ext == ".yaml":
		with open(filename, "w") as f:
			yaml.safe_dump(data, f, default_flow_style=False, encoding='utf-8',  allow_unicode=True)
	else:
		raise ValueError("file should end in .json or .yaml (not %s)" % ext)

def read_json_or_yaml(filename):
	base, ext = os.path.splitext(filename)
	if ext == ".json":
		with open(filename, "r") as f:
			return json.load(f) or {}
	elif ext == ".yaml":
		with open(filename, "r") as f:
			return yaml.load(f) or {}
	else:
		raise ValueError("file should end in .json or .yaml (not %s)" % ext)

# from http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
import collections
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_representer(dumper, data):
    return dumper.represent_dict(data.iteritems())

def dict_constructor(loader, node):
    return collections.OrderedDict(loader.construct_pairs(node))

yaml.add_representer(collections.OrderedDict, dict_representer, yaml.SafeDumper)
yaml.add_constructor(_mapping_tag, dict_constructor, yaml.SafeLoader)

import psutil
def check_memory_usage(bytes_needed, confirm):
	if bytes_needed > psutil.virtual_memory().available:
		if bytes_needed < (psutil.virtual_memory().available +psutil.swap_memory().free):
			text = "Action requires %s, you have enough swap memory available but it will make your computer slower, do you want to continue?" % (filesize_format(bytes_needed),)
			return confirm("Memory usage issue", text)
		else:
			text = "Action requires %s, you do not have enough swap memory available, do you want try anyway?" % (filesize_format(bytes_needed),)
			return confirm("Memory usage issue", text)
	return True

import six

def ensure_string(string_or_bytes, encoding="utf-8"):
	if isinstance(string_or_bytes, six.string_types):
		return string_or_bytes
	else:
		return string_or_bytes.decode(encoding)