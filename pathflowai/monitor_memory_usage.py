import GPUtil
import psutil
import time
from threading import Thread
import pandas as pd
import argparse

class Monitor(Thread):
	def __init__(self, start_time, delay, end_time):
		super(Monitor, self).__init__()
		self.stopped = False
		self.start_time = start_time
		self.end_time = end_time
		self.delay = delay # Time between calls to GPUtil
		self.records = []
		self.start()

	def run(self):
		time_from_start = 0.
		while time_from_start <= self.end_time:
			memory = psutil.virtual_memory()
			stats = {"gpu.{}.memory.used".format(gpu.id):gpu.memoryUsed for gpu in GPUtil.getGPUs()}
			stats['cpu.utilization'] = psutil.cpu_percent()
			current_time = time.time()
			stats['current.time'] = current_time
			time_from_start = current_time - self.start_time
			stats['system.memory.used'] = memory.used
			stats['system.memory.used.percent'] = memory.percent
			stats['elapsed.time'] = time_from_start
			self.records.append(stats)
			time.sleep(self.delay)
		self.stop()

	def stop(self):
		self.stopped = True

	def return_records(self):
		return pd.DataFrame(self.records)

def get_usage(total_time, delay_time, records_output_csv):

	start_time = time.time()

	monitor = Monitor(start_time, delay_time, total_time)

	monitor.run()

	while not monitor.stopped:
		time.sleep(delay_time)

	records = monitor.return_records()

	records.to_csv(records_output_csv)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Monitor Usage over Time Interval.')
	parser.add_argument('--records_output_csv', default='records.csv',type=str,help='Where to store records.')
	parser.add_argument('--total_time', default=1., type=float, help='Total time to monitor for in minutes.')
	parser.add_argument('--delay_time', default=1., type=float, help='Time between samples, in seconds.')
	args = parser.parse_args()
	records_output_csv = args.records_output_csv
	total_time = args.total_time # in minutes
	total_time*= 60. # convert to seconds
	delay_time = args.delay_time # in seconds
	get_usage(total_time, delay_time, records_output_csv)
