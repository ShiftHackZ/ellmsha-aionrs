//! Host-owned guidance is projected into requests, never conversation history.

use aion_types::message::{ContentBlock, Message, Role};

pub(crate) struct RequestGuidance {
    text: String,
    message_index: usize,
}

impl RequestGuidance {
    pub(crate) fn new(text: String, message_index: usize) -> Self {
        Self { text, message_index }
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn after_compaction(&mut self) {
        // Autocompact replaces all history with a boundary and task summary.
        self.message_index = 1;
    }

    pub(crate) fn project(&self, messages: &mut Vec<Message>) {
        let block = ContentBlock::Text {
            text: self.text.clone(),
        };
        if let Some(message) = messages.get_mut(self.message_index)
            && message.role == Role::User
        {
            message.content.insert(0, block);
        } else {
            // Commands/empty history have no new task message to prefix.
            messages.push(Message::new(Role::User, vec![block]));
        }
    }
}
