#Source: https://stackoverflow.com/questions/53286181/typeerror-book-takes-1-positional-argument-but-2-were-given
class main_menu(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pic = self.ids["pic"]
        self.pic1 = self.ids["pic1"]
        self.ac=self.ids["ac"]
        self.sd=self.ids["sd"]
        print(phone)

        #self.ac.text=self.user_text.text
        #print(self.l.user_text.text,"jkckhjkfgh")

        self.lst = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
        self.sql="select * from snm"
        cursor.execute(self.sql)
        self.result = cursor.fetchall()


        self.pic.add_widget(Button(text = "               {}     ".format(self.result[0][0])  + '\n' +
                                        "                  {} , {} ".format(self.result[0][2],self.result[0][3])
                                        + "\n\n               Seating Capacity  :"
                                        + "               {}".format(self.result[0][1])
                                   ,font_size=30
                                   ,font_name="font1"
                                   ,background_normal=self.lst[0]
                                   ,on_press=self.book
                                   ))




        self.pic.add_widget(Button(text="               {}     ".format(self.result[1][0]) + '\n' +
                                        "                  {} , {} ".format(self.result[1][2], self.result[1][3])
                                        + "\n\n               Seating Capacity  :"
                                        + "               {}".format(self.result[1][1])
                                   ,font_size=30
                                   ,font_name="font1"
                                   ,background_normal=self.lst[1]
                                   ,on_press=self.book
                                   ))




        self.pic.add_widget(Button(text="               {}     ".format(self.result[2][0]) + '\n' +
                                        "                  {} , {} ".format(self.result[2][2], self.result[2][3])
                                        + "\n\n               Seating Capacity  :"
                                        + "              {}".format(self.result[2][1])
                                   ,font_size=30
                                   ,font_name="font1"
                                   ,background_normal=self.lst[2]
                                   ,on_press=self.book
                                   ))




        self.pic.add_widget(Button(text="               {}     ".format(self.result[3][0]) + '\n' +
                                        "                  {} , {} ".format(self.result[3][2], self.result[3][3])
                                        + "\n\n               Seating Capacity  :"
                                        + "               {}".format(self.result[3][1])
                                   ,font_size=30
                                   ,font_name="font1"
                                   ,background_normal=self.lst[3]
                                   ,on_press=self.book
                                   ))
        self.h = 4
        self.pic.size = (100, self.h *600)
    def book(self):
        sm.current="book_ticket"
class PopUp(Popup):
    def set(self):
        self.title = 'Wrong Login Details'
        self.content = Label(text = 'You have entered wrong Login Details',font_size=30)
        self.size_hint = (None,None)
        self.size = (600,250)
class book_ticket(Screen):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
t1=signin(name='signin')
t2=details(name='details')
t3=main_menu(name="main_menu")
t4=book_ticket(name="book_ticket")
print(phone)
print(type(t1))
            #print(t1.user_text.text,phone,23)
            #create local login applic def Start(self):
sm = ScreenManager()
sm.add_widget(t1)
sm.add_widget(t2)
sm.add_widget(t3)
sm.add_widget(t4)
class main(App):
    def build(self):
        return main_menu()
main().run()