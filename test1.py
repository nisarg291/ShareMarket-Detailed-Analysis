

class Node:

    def __init__(self,data):
        self.data = data
        self.next = None
    
class LinkedList:

    def __init__(self):
        self.root=None

    def append(self, data):
        n=self.root
        if n==None:
            self.root=Node(data)
        else:
            while n.next!=None:
                n=n.next
            
            n.next=Node(data)

    def removeFirst(self):
        n=self.root
        self.root=n.next
        n.next=None

    def removeLast(self):
        n=self.root
        if n.next==None:
            n=None
        elif n!=None and n.next!=None:
            while n.next!=None and n.next.next!=None:
                n=n.next
            n.next=None
        
        
            

    def printList(self):
        n=self.root
        while n.next!=None:
            print(n.data, "->")
            n=n.next
        print(n.data)


def main():
    list = LinkedList()
    list.append(10)
    list.append(20)
    list.append(30)
    list.append(40)
    list.printList()
    list.removeFirst()
    list.removeLast()
    list.printList()

    
main()
