import os
import json
import cv2
import random as r
import numpy as np

def draw_connection(image, cnt1, cnt2, layout = {"radius": 4, "color": (0, 255, 0), "line_color": (0, 255, 0), "thickness": 2}):

    # Draw the two circles
    cv2.circle(image, (cnt1[0], cnt1[1]), layout["radius"], layout["color"], cv2.FILLED)
    cv2.circle(image, (cnt2[0], cnt2[1]), layout["radius"], layout["color"], cv2.FILLED)

    # Draw a line connecting the centers of the circles
    cv2.line(image, cnt1, cnt2, layout["line_color"], layout["thickness"])

    return image


class SequenceIntake:

    def __init__(self, data):
        self.data = data
        self.instances = []
    
    def add_instance(self, st, en):
        self.instances.append(SequenceInstance(self.data, st, en-st))

    def insert_instance(self, instance_data, start, limit_size = False):
        original_shape = self.data.shape
        end = (instance_data.shape[0] + start)
        
        new_data = np.zeros((self.data.shape[0] + instance_data.shape[0], self.data.shape[1], self.data.shape[2]))
        new_data[:start] = self.data[:start]
        new_data[start:end] = instance_data[:]
        new_data[end:] = self.data[start:]

        for instance in self.instances:
            if instance.start > start:
                instance.start = instance.start + instance_data.shape[0]

        if limit_size:
            self.data = new_data[:self.data.shape[0]]
        else:        
            self.data = np.zeros(new_data.shape)
            self.data = new_data[:]
        self.add_instance(start, start + instance_data.shape[0])

    def find_similar_frame(self, frame_to_match, indices = [10, 11, 12, 13, 14, 15, 16]):
        compare_data = self.data[:, indices]
        compare_data = compare_data.reshape(compare_data.shape[0], len(indices) * compare_data.shape[2])
        frame_info = frame_to_match[indices]
        frame_info = frame_info.reshape(frame_info.shape[0] * frame_info.shape[1])
        vectors = compare_data - frame_info
        dist = np.linalg.norm(vectors[:], axis = 1)
        similar_frames = np.argsort(dist)
        return similar_frames
    
    def visualize_frame(self, n, width = 400, height = 400):

        info = self.data[n]
        img = np.zeros((width, height, 3)).astype(np.uint8)

        # take out offset for bounding box
        info[2:, 0] = info[2:, 0] * height/2
        info[2:, 1] = info[2:, 1] * width/2
        info = info.astype(np.uint8)

        img = draw_connection(img, info[2], info[3])
        img = draw_connection(img, info[2], info[4])
        img = draw_connection(img, info[3], info[4])

        img = draw_connection(img, info[3], info[5])
        img = draw_connection(img, info[4], info[6])
        img = draw_connection(img, info[5], info[7])
        img = draw_connection(img, info[6], info[8])

        img = draw_connection(img, info[4], info[10])
        img = draw_connection(img, info[3], info[9])
        img = draw_connection(img, info[9], info[10])

        img = draw_connection(img, info[10], info[12])
        img = draw_connection(img, info[9], info[11])
        img = draw_connection(img, info[12], info[14])
        img = draw_connection(img, info[11], info[13])
        img = draw_connection(img, info[14], info[16])
        img = draw_connection(img, info[13], info[15])

        return img
    


class SequenceInstance:

    def __init__(self, data, start, span):
        self.data = data
        self.start = start
        self.span = span

    def getData(self):
        return self.data[self.start:self.start + self.span]

    def getStartFrame(self):
        return self.data[self.start]
    
    def getEndFrame(self):
        return self.data[self.start + self.span]

    def copy(self):
        new_data = np.zeros(self.data.shape) + self.data[:]
        return SequenceInstance(new_data, self.start, self.span)

    def reverse(self):
        self.data = np.flip(self.data, axis=0)
        self.start = self.data.shape[0] - (self.start + self.span)

    def jiggle(self, frames, indices, axi = [0, 1], fraction = .1):
        for a in axi:
            jiggle = np.random.rand(len(frames), len(indices), 1) * fraction
            self.data[np.ix_(frames, indices, [a])] = self.data[np.ix_(frames, indices, [a])] + jiggle
    
    def jiggle_knees(self, factor):
        frames = np.arange(self.span) + self.start
        self.jiggle(frames, [11, 12], [0, 1], fraction = factor)
    
    def jiggle_hands(self, factor):
        frames = np.arange(self.span) + self.start
        frames = np.where(self.data[np.ix_(frames, [7, 8], [1])] < .5)[0]
        print(frames)
        self.jiggle(list(frames), [7,8], [0, 1], fraction = factor)
    
    def jiggle_elbows(self, factor):
        frames = np.arange(self.span) + self.start
        self.jiggle(frames, [5,6], [0, 1], fraction = factor)
    
    def mirror(self, isolate_to_instance = False):
        mirrored = np.zeros(self.data.shape) + self.data[:]
        axis = np.zeros(mirrored.shape) + mirrored[:]
        axis[:,:,0] = .5
        vectors = axis - mirrored
        mirrored = mirrored + vectors * 2
        if isolate_to_instance:
            self.data[self.start:self.start+self.span] = mirrored[self.start:self.start+self.span]
        else:
            self.data[:] =  mirrored



def main():

    input_size = 1100

    data = np.load("./input/features/SixStepPeter.npy")
    myInstances = SequenceIntake(data)
    myInstances.add_instance(84, 140)
    myInstances.add_instance(195, 250)
    myInstances.add_instance(250, 305)

    allIntake = SequenceIntake(np.load("./input/features/IMG_3353.npy")[:input_size])
    outputs = []
    instances = []

    for i in range(len(myInstances.instances)):
        mirrored_instance = myInstances.instances[i].copy()
        reversed_instance = myInstances.instances[i].copy()
        mirrored_reversed_instance = myInstances.instances[i].copy()

        mirrored_instance.mirror()
        reversed_instance.reverse()

        instances.append(myInstances.instances[i])
        instances.append(mirrored_instance)
        instances.append(reversed_instance)
        instances.append(mirrored_reversed_instance)
        

    for entry in instances:

        possible_starts = allIntake.find_similar_frame(entry.getStartFrame())
        possible_ends = allIntake.find_similar_frame(entry.getEndFrame())

        possible_starts = possible_starts[np.where(possible_starts < input_size - entry.span - 100)[0]]
        possible_ends = possible_ends[np.where(possible_ends < input_size - 100)]

        if len(possible_starts) > 0 and len(possible_ends) > 0:

            for j in range(2):

                start = possible_starts[j]
                end = possible_ends[j]
                
                copy_data = np.zeros(allIntake.data.shape)[:input_size]
                copy_data[:] = allIntake.data[:input_size]
                new_intake = SequenceIntake(copy_data)
                new_intake.insert_instance(entry.getData(), start)
                new_end = start + entry.span
                new_intake.data[new_end:new_end + 100] = allIntake.data[end:end + 100]
                new_intake.data = new_intake.data[:input_size]

                outputs.append(new_intake)
    
    count = 0
    for o in outputs:
        np.save(f"./instances/data/{count}.npy", o.data)
        labels = {"instances":[]}
        for i in range(len(o.instances)):
            labels["instances"].append({"start": int(o.instances[i].start), "span": int(o.instances[i].span)})
        with open(f"./instances/labels/{count}.json", 'w') as json_file:
            json.dump(labels, json_file, indent=4)
        for i in range(o.data.shape[0]):
            img = o.visualize_frame(i)
            if os.path.exists(f"./instances/visualized/{count}") == False:
                os.mkdir(f"./instances/visualized/{count}")
            cv2.imwrite(f"./instances/visualized/{count}/frame_{i}.jpg", img)
        count = count + 1
        

    
main()