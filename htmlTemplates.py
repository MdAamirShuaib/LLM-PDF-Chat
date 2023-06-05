css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://media.istockphoto.com/id/1060696342/vector/robot-icon-chat-bot-sign-for-support-service-concept-chatbot-character-flat-style.jpg?s=170667a&w=0&k=20&c=jQkzD_qek6cWxmyCh4V04tb9O2FOdC0Br2ycJ4QdTyk=" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/021/548/095/original/default-profile-picture-avatar-user-avatar-icon-person-icon-head-icon-profile-picture-icons-default-anonymous-user-male-and-female-businessman-photo-placeholder-social-network-avatar-portrait-free-vector.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
"""


footer = """
<div style="position: fixed; left: 400px; bottom: 0; width: 100%; background-color: #0e1117; color: white; text-align: left;">
    <a href="https://github.com/MdAamirShuaib">Github</a> &nbsp;
    <a href="https://www.linkedin.com/in/aamirshuaib/">LinkedIn</a> &nbsp;
    <a href="https://mohammed-aamir-shuaib.vercel.app/">Portfolio</a> &nbsp;
    <a href="mailto:aamirshuaib0@gmail.com">Email</a>
    <p>Made by Mohammed Aamir Shuaib</p>
</div>
"""

completed = """
<div style="display:flex; align-items:center;"><span style="color:green">\u2713 {{text}}</span></div>
"""
