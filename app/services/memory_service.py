from supabase import create_client
import os

SYSTEM_PROMPT = (
    """
    Summarize the conversation in under 100 words. Keep only user preferences. If an existing summary exists update it with respect to the new messages.
    """
)

USER_PROMPT = (
   """
    Summarize the conversation.
    {conversation}
    """
)

class MemoryService:
    def __init__(self, llm):
           
        self.client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        self.llm = llm 
        
    def save_message(self, user_id, session_id, role, content, images=None, citations=None):
        
        self.client.table("internal_kb").insert({
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "images" : images or [],
            "citations" : citations or []
        }).execute()

        # update session timestamp
        self.client.table("sessions").update({
            "updated_at": "now()"
        }).eq("id", session_id).execute()

    def get_history(self, user_id, session_id, limit=6):
        summary = self.get_summary_record(user_id, session_id)["content"]
        response = self.client.table("internal_kb") \
            .select("role, content") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .neq("role", "summary") \
            .order("created_at", desc=False) \
            .limit(limit) \
            .execute()
        history = response.data
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in history
        ])
        if summary:
            return summary + history_text
        else:
            return history_text

    
    def get_summary_record(self, user_id, session_id):
        summary_row = self.client.table("internal_kb") \
        .select("id, content, created_at") \
        .eq("user_id", user_id) \
        .eq("session_id", session_id) \
        .eq("role", "summary") \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()

        summary = summary_row.data[0] if summary_row.data else None
        return summary

    
    def generate_summary(self, input):
        user_prompt = USER_PROMPT.format(
            conversation=input
        )

        response = self.llm.invoke([
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
        ])

        return response.content


    
    def maybe_summarize(self, user_id, session_id):
        
        summary_record = self.get_summary_record(user_id, session_id)
        query = self.client.table("internal_kb") \
        .select("id, role, content, created_at") \
        .eq("user_id", user_id) \
        .eq("session_id", session_id) \
        .neq("role", "summary") \
        .order("created_at", desc=False)

        if summary_record:
            query = query.gt("created_at", summary_record["created_at"])

        messages = query.execute().data or []

        if len(messages) <= 6:
            return

        to_summarize = messages[:-6]

        new_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in to_summarize
        )

        if summary_record:
            input_text = f"""
        Existing summary:
        {summary_record['content']}

        New messages:
        {new_text}
        """
        else:
            input_text = new_text

        new_summary = self.generate_summary(input_text)

        # insert new summary
        new_row = self.client.table("internal_kb").insert({
            "user_id": user_id,
            "session_id": session_id,
            "role": "summary",
            "content": new_summary
        }).execute()

        # delete old summary using ID (optional)
        if summary_record:
            self.client.table("internal_kb") \
                .delete() \
                .eq("id", summary_record["id"]) \
                .execute()

    def get_history_ui(self, user_id, session_id):
        response = self.client.table("internal_kb") \
            .select("role, content, images, citations") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .neq("role", "summary") \
            .order("created_at") \
            .execute()
        return [{
            "role": row['role'],
            "content": row['content'],
            "images": row['images'],
            "citations": row['citations']
            }
            for row in response.data
            ]
    
    def clear_history(self, user_id, session_id):
        response = self.client.table("internal_kb") \
            .delete() \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .execute()
        response = self.client.table("sessions") \
            .delete() \
            .eq("id", session_id) \
            .execute()

        return response
    
        
    def create_session_if_not_exist(self, user_id, session_id):
        response = self.client.table("sessions") \
            .select("id") \
            .eq("id", session_id) \
            .execute()
        
        if not response.data:
            self.client.table("sessions").insert({
                "id": session_id,
                "user_id":user_id
            }).execute()

    def get_sessions(self, user_id):
        res = self.client.table("sessions") \
            .select("id, updated_at") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .execute()

        return [
            {
                "session_id": row["id"]
            }
            for row in res.data
        ]


