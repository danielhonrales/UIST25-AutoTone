import tkinter as tk
import math

class EmotionWheelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Wheel")
        self.root.geometry("600x600")
        
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack()

        self.center = (300, 300)
        self.radius = 200
        self.emotion_radius = 120
        self.sections = ["Neutral", "Happy", "Angry", "Sad", "Surprised"]
        self.emotion_scores = [0, 0, 0, 0]  # Scores for Happy, Sad, Angry, Surprised
        self.current_section = "Neutral"
        self.angle = 0
        
        self.is_dragging = False
        self.toggle_mode = False

        self.create_wheel()
        self.create_dial_indicator()
        self.create_dot()
        self.create_mode_toggle_button()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def create_wheel(self):
        # Create color sections
        section_angles = 360 / (len(self.sections) - 1)  # Exclude neutral
        section_colors = ["#D3D3D3", "#FFFF00", "#FF6347", "#1E90FF", "#FFD700"]

        # Draw sections
        for i, section in enumerate(self.sections[1:]):  # Skip "Neutral"
            angle_start = i * section_angles
            angle_end = (i + 1) * section_angles
            self.draw_section(angle_start, angle_end, section_colors[i + 1], section)
        
        # Draw neutral section as a smaller circle at the center
        self.create_neutral_section()

    def draw_section(self, start_angle, end_angle, color, label):
        # Convert angle to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        
        # Create the section (Arc)
        self.canvas.create_arc(self.center[0] - self.radius, self.center[1] - self.radius,
                               self.center[0] + self.radius, self.center[1] + self.radius,
                               start=start_angle, extent=end_angle - start_angle,
                               fill=color, outline="black", width=2)
        
        # Draw label outside the circle
        angle_mid = (start_angle + end_angle) / 2
        label_x = self.center[0] + (self.radius + 30) * math.cos(math.radians(angle_mid))  # Increased distance
        label_y = self.center[1] + (self.radius + 30) * math.sin(math.radians(angle_mid))  # Increased distance
        
        self.canvas.create_text(label_x, label_y, text=label, font=("Arial", 12, "bold"), fill="black")

    def create_neutral_section(self):
        # Draw the neutral section as a smaller circle in the center
        self.canvas.create_oval(self.center[0] - 50, self.center[1] - 50,
                                self.center[0] + 50, self.center[1] + 50, 
                                fill="#B0B0B0", outline="black")
        self.canvas.create_text(self.center[0], self.center[1], text="Neutral", 
                                font=("Arial", 12, "bold"), fill="black")

    def create_dial_indicator(self):
        # Create the dial indicator (line)
        self.dial_indicator = self.canvas.create_line(self.center[0], self.center[1], 
                                                      self.center[0], self.center[1] - self.radius, 
                                                      width=3, fill="black")
        self.update_indicator(self.angle)

    def update_indicator(self, angle):
        # Update the dial indicator position
        x = self.center[0] + self.radius * math.sin(math.radians(angle))
        y = self.center[1] - self.radius * math.cos(math.radians(angle))
        self.canvas.coords(self.dial_indicator, self.center[0], self.center[1], x, y)
        self.update_section(angle)

    def update_section(self, angle):
        # Determine which section the dial is pointing to
        section_angles = 360 / (len(self.sections) - 1)  # Exclude neutral
        section_index = int((angle + section_angles / 2) // section_angles)
        if section_index == 0 or section_index + 1 >= len(self.sections):
            self.current_section = "Neutral"
        else:
            self.current_section = self.sections[section_index + 1]

    def on_click(self, event):
        # Check if click is inside the circle and set dragging flag
        if (event.x - self.center[0])**2 + (event.y - self.center[1])**2 <= self.radius**2:
            self.is_dragging = True
            self.angle = math.degrees(math.atan2(event.y - self.center[1], event.x - self.center[0])) % 360
            self.update_indicator(self.angle)

    def on_drag(self, event):
        if self.is_dragging:
            # Calculate the new angle based on the mouse position
            self.angle = math.degrees(math.atan2(event.y - self.center[1], event.x - self.center[0])) % 360
            self.update_indicator(self.angle)

    def on_release(self, event):
        # Reset dragging flag
        self.is_dragging = False

    def create_dot(self):
        # Create a dot that moves based on emotion scores
        self.dot = self.canvas.create_oval(self.center[0] - 5, self.center[1] - 5,
                                           self.center[0] + 5, self.center[1] + 5, fill="black")
        self.update_dot()

    def update_dot(self):
        # Calculate position of the dot based on emotion scores
        total_score = sum(self.emotion_scores)
        angle_offset = 360 / 4
        dot_x = self.center[0]
        dot_y = self.center[1]
        
        for i, score in enumerate(self.emotion_scores):
            if score > 0:
                angle = angle_offset * (i + 1)
                radius = self.emotion_radius * (score / total_score if total_score != 0 else 0)
                dot_x += radius * math.cos(math.radians(angle))
                dot_y += radius * math.sin(math.radians(angle))
        
        self.canvas.coords(self.dot, dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5)

    def create_mode_toggle_button(self):
        # Create a toggle button for switching modes
        self.toggle_button = tk.Button(self.root, text="Switch Mode", command=self.toggle_mode_function)
        self.toggle_button.pack(pady=10)

    def toggle_mode_function(self):
        # Switch between modes
        self.toggle_mode = not self.toggle_mode

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionWheelApp(root)
    root.mainloop()
