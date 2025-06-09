"""
Vector Database Schema Definition
Defines the metadata schema for JIRA tickets and documents stored in Chroma
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import datetime


class SourceType(Enum):
    """Data source types"""
    CSV = "csv"
    S3 = "s3"
    CONFLUENCE = "confluence"


class TicketStatus(Enum):
    """Common JIRA ticket statuses"""
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    DONE = "Done"
    CLOSED = "Closed"
    RESOLVED = "Resolved"
    TO_DO = "To Do"
    TESTING = "Testing"
    REVIEW = "Review"


class TicketPriority(Enum):
    """JIRA ticket priorities"""
    LOWEST = "Lowest"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    HIGHEST = "Highest"
    CRITICAL = "Critical"


class IssueType(Enum):
    """JIRA issue types"""
    BUG = "Bug"
    STORY = "Story"
    TASK = "Task"
    EPIC = "Epic"
    SUBTASK = "Sub-task"
    IMPROVEMENT = "Improvement"
    NEW_FEATURE = "New Feature"


@dataclass
class DocumentMetadata:
    """Schema for document metadata in vector store"""
    
    # Core identification
    ticket_id: str
    source: str  # SourceType value
    
    # JIRA specific fields
    summary: str = ""
    status: str = ""
    priority: str = ""
    issue_type: str = ""
    assignee: str = ""
    reporter: str = ""
    
    # Timestamps (as strings for Chroma compatibility)
    created_date: str = ""
    updated_date: str = ""
    resolved_date: str = ""
    
    # Additional fields
    component: str = ""
    version: str = ""
    fix_version: str = ""
    labels: str = ""  # Comma-separated labels
    
    # Confluence specific (when applicable)
    space_key: str = ""
    page_title: str = ""
    
    # Processing metadata
    chunk_index: int = 0
    total_chunks: int = 1
    processed_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Chroma storage"""
        return {
            "ticket_id": self.ticket_id,
            "source": self.source,
            "summary": self.summary[:200],  # Limit summary length
            "status": self.status,
            "priority": self.priority,
            "issue_type": self.issue_type,
            "assignee": self.assignee,
            "reporter": self.reporter,
            "created_date": self.created_date,
            "updated_date": self.updated_date,
            "resolved_date": self.resolved_date,
            "component": self.component,
            "version": self.version,
            "fix_version": self.fix_version,
            "labels": self.labels,
            "space_key": self.space_key,
            "page_title": self.page_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "processed_at": self.processed_at
        }
    
    @classmethod
    def from_jira_row(cls, row: Dict[str, Any], source: str = "csv", chunk_index: int = 0, total_chunks: int = 1) -> 'DocumentMetadata':
        """Create metadata from JIRA CSV row"""
        return cls(
            ticket_id=str(row.get("Issue key", "Unknown")),
            source=source,
            summary=str(row.get("Summary", ""))[:200],
            status=str(row.get("Status", "")),
            priority=str(row.get("Priority", "")),
            issue_type=str(row.get("Issue Type", "")),
            assignee=str(row.get("Assignee", "")),
            reporter=str(row.get("Reporter", "")),
            created_date=str(row.get("Created", "")),
            updated_date=str(row.get("Updated", "")),
            resolved_date=str(row.get("Resolved", "")),
            component=str(row.get("Component/s", "")),
            version=str(row.get("Affects Version/s", "")),
            fix_version=str(row.get("Fix Version/s", "")),
            labels=str(row.get("Labels", "")),
            chunk_index=chunk_index,
            total_chunks=total_chunks
        )
    
    @classmethod
    def from_confluence_doc(cls, doc: Any, space_key: str, chunk_index: int = 0, total_chunks: int = 1) -> 'DocumentMetadata':
        """Create metadata from Confluence document"""
        return cls(
            ticket_id=f"confluence_{space_key}_{hash(doc.page_content) % 10000}",
            source="confluence",
            summary=str(doc.metadata.get("title", ""))[:200],
            space_key=space_key,
            page_title=str(doc.metadata.get("title", "")),
            chunk_index=chunk_index,
            total_chunks=total_chunks
        )


class VectorStoreSchema:
    """Schema management for the vector store"""
    
    @staticmethod
    def get_collection_metadata() -> Dict[str, Any]:
        """Get collection-level metadata configuration"""
        return {
            "description": "Enterprise RAG JIRA Tickets and Documentation",
            "schema_version": "1.0",
            "created_at": datetime.datetime.now().isoformat(),
            "fields": {
                "ticket_id": "string - Unique ticket identifier",
                "source": "string - Data source (csv, s3, confluence)",
                "summary": "string - Ticket or document summary (max 200 chars)",
                "status": "string - JIRA ticket status",
                "priority": "string - JIRA ticket priority",
                "issue_type": "string - JIRA issue type",
                "assignee": "string - Assigned person",
                "reporter": "string - Reporter/creator",
                "created_date": "string - Creation timestamp",
                "updated_date": "string - Last update timestamp",
                "resolved_date": "string - Resolution timestamp",
                "component": "string - Component/module",
                "version": "string - Affected version",
                "fix_version": "string - Target fix version",
                "labels": "string - Comma-separated labels",
                "space_key": "string - Confluence space key",
                "page_title": "string - Confluence page title",
                "chunk_index": "int - Document chunk index",
                "total_chunks": "int - Total chunks for document",
                "processed_at": "string - Processing timestamp"
            }
        }
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """Validate metadata against schema"""
        required_fields = ["ticket_id", "source"]
        
        for field in required_fields:
            if field not in metadata:
                return False
        
        # Validate source type
        if metadata["source"] not in [e.value for e in SourceType]:
            return False
        
        return True
    
    @staticmethod
    def get_filter_examples() -> Dict[str, Dict[str, Any]]:
        """Get example filters for common queries"""
        return {
            "high_priority_bugs": {
                "priority": {"$in": ["High", "Highest", "Critical"]},
                "issue_type": "Bug",
                "status": {"$nin": ["Done", "Closed", "Resolved"]}
            },
            "recent_tickets": {
                "source": {"$in": ["csv", "s3"]},
                "created_date": {"$gte": "2024-01-01"}
            },
            "specific_assignee": {
                "assignee": "john.doe@company.com",
                "status": {"$nin": ["Done", "Closed"]}
            },
            "confluence_docs": {
                "source": "confluence",
                "space_key": "ENGINEERING"
            }
        } 